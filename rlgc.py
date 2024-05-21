#!/usr/bin/env python

# Richardson-Lucy deconvolution code, using gradient consensus to stop iterations locally
# James Manton, 2023
# jmanton@mrc-lmb.cam.ac.uk
#
# Developed in collaboration with Andy York (Calico), Jan Becker (Oxford) and Craig Russell (EMBL EBI)
# Minor edits by Scott Brooks (Warwick) s.brooks.2@warwick.ac.uk

import numpy as np
import cupy as cp
import ast
import timeit
import tifffile
import argparse
import json

# python rlgc.py --input "/mnt/Raid_partition_1/internal_tmp/brooks/dockerdir/MantonData/2024-05-13_montage7Capture10crop448_t1.tif" --psf "/mnt/Raid_partition_1/internal_tmp/brooks/dockerdir/MantonData/448SamplePSF2_cropped.tif" --output "/mnt/Raid_partition_1/internal_tmp/brooks/dockerdir/MantonData/2024-05-13_montage7Capture10crop448_t1_rlgc_iter20.tif" --rl_output "/mnt/Raid_partition_1/internal_tmp/brooks/dockerdir/MantonData/debug_iterations20.tif" --max_iters 20


rng = np.random.default_rng()

# TODO: additional options add redundant code where 

def main():
    # Get input arguments
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type = str, required = True)
    # should be list of strings
    parser.add_argument('--psf', type = str, required = True)
    parser.add_argument('--output', type = str, required = True)
    parser.add_argument('--max_iters', type = int, default = 10)
    parser.add_argument('--reblurred', type = str, required = False)
    parser.add_argument('--process_psf', type = int, default = 1)
    parser.add_argument('--rl_output', type = str, required = True) #originally False, return this to False later
    parser.add_argument('--limit', type = float, default = 0.01)
    parser.add_argument('--iters_output', type = str, required = False)
    parser.add_argument('--rl_iters_output', type = str, required = False)
    parser.add_argument('--updates_output', type = str, required = False)
    parser.add_argument('--blur_consensus', type = int, default = 1)
    args = parser.parse_args()

    # # Log which files we're working with and the number of iterations
    # print('')
    # print('Input file: %s' % args.input)
    # print('Input shape: %s' % (image.shape, ))
    # print('PSF files: %s' % args.psf)
    # print('PSF shape: %s' % (psf_temp.shape, ))
    # print('Output file: %s' % args.output)
    # print('Maximum number of iterations: %d' % args.max_iters)
    # print('PSF processing: %s' % args.process_psf)
    # print('')

    # Print raw input arguments for debugging
    print(f"Raw PSF input: {args.psf}")

    # Split the comma-separated string into a list
    psf_list = args.psf.split(',')

    # Ensure it's a list
    if not isinstance(psf_list, list):
        raise ValueError("The argument for --psf should be a comma-separated list of strings.")


    with tifffile.TiffFile(args.input) as tif:
        full_image = tif.asarray()
        # metadata = tif.shaped_metadata
        imagej_metadata = tif.imagej_metadata if tif.is_imagej else {}
    print(args.input)
    

    image_shape = get_image_shape(full_image, psf_list)
    
    
    # Load PSFs
    psfs = load_psfs(args, psf_list, image_shape)
    tifffile.imwrite("processedpsfs.tif", psfs.get(), bigtiff=True)
    
    recon, recon_rl = triage(args, full_image, psfs)

    with tifffile.TiffWriter(args.output, imagej=True) as tif_writer:
        combined_metadata = {'axes': 'TZCYX'}
        combined_metadata.update(imagej_metadata)
        tif_writer.write(recon, metadata=combined_metadata)

    
    print(recon.shape)
    
    # Save RL output if argument given
    # if args.rl_output is not None:
    #     tifffile.imwrite(args.rl_output, recon_rl, bigtiff=True, metadata={'axes': 'TZCYX'}, imagej_metadata=metadata)

    # not implemeted for 5D
	# # Save full iterations if argument given
	# if (args.iters_output is not None):
	# 	tifffile.imwrite(args.iters_output, iters[0:num_iters, :, :, :], bigtiff=True)

    # not implemeted for 5D
	# # Save full RL iterations if argument given
	# if (args.rl_iters_output is not None):
	# 	tifffile.imwrite(args.rl_iters_output, rl_iters[0:num_iters, :, :, :], bigtiff=True)

	# not implemeted for 5D
	# # Save full updates if argument given
	# if (args.updates_output is not None):
	# 	tifffile.imwrite(args.updates_output, updates, bigtiff=True)
def triage(args, full_image, psfs):
    # if there is only 1 timepoint just run original code, else assume ztcyx
    
    # TODO: come up with something better than enumerating possibilities + channel lazy 2 GPU processing
    if full_image.ndim <= 3:
        return rlgc(args, full_image, psfs[0], 0, 0)
    # TODO: MULTIchannel we need to make the shape
    elif full_image.ndim == 4:
        if len(psfs) == 1:
            return rlgc_4D(args, full_image, psfs[0], 0)
        else:
            return rlgc_4D_multichannel(args, full_image, psfs)
    elif full_image.ndim == 5:
        # TODO: Should this also have some alignment if psfs are very different?
        return rlgc_5D(args, full_image, psfs)

def get_image_shape(full_image, psf_list):

    # assume tzcyx or zcyx
    shape = full_image.shape
    if len(shape) == 4 and len(psf_list) == 2:
        first_element = shape[0]
    else:
        first_element = shape[1]

    last_two_elements = shape[-2:]
    
    # Combine into a new tuple
    return (first_element, *last_two_elements)

def load_psfs(args, psf_list, image_shape):
    
    # this is a little inefficient, but avoiding lists, should we store the first read?
    # psf_temp = tifffile.imread(psf_list[0])e
    new_shape = (len(psf_list),) + image_shape
    psfs = cp.zeros(new_shape, dtype=cp.float32)
    
    for i, psf_path in enumerate(psf_list):
        # Load and pad PSF if necessary
        print(psf_path)
        psf_temp = tifffile.imread(psf_path)
    
        # Add new z-axis if we have 2D data
        if psf_temp.ndim == 2:
            psf_temp = np.expand_dims(psf_temp, axis=0)
    
        if (args.process_psf):
            print("Processing PSF...")
            # Take upper left 16x16 pixels to estimate noise level and create appropriate fake noise
            noisy_region = psf_temp[0:16, 0:16, 0:16]
            background = np.random.normal(np.mean(noisy_region), np.std(noisy_region), image_shape)
        
        psf = np.zeros(image_shape)
    
        # Place PSF is the top left corner of an array with the same shape of the original image
        psf[:psf_temp.shape[0], :psf_temp.shape[1], :psf_temp.shape[2]] = psf_temp
    
        # This looks like it centers the PSF, but for larger PSFs in comparison to image size will not centre well, without this it doesnt work, what does it do?
        for axis, axis_size in enumerate(psf_temp.shape):
        	psf = np.roll(psf, -int(axis_size / 2), axis=axis)
    
        # Calculate the shift values to center psf_temp in psf
        shifts = [-(psf_temp.shape[i] // 2) + (psf.shape[i] // 2) for i in range(len(psf_temp.shape))]
        
        # # Roll psf to center psf_temp
        # for axis, shift in enumerate(shifts):
        #     psf = np.roll(psf, shift, axis=axis)
            

    
    
        # remove any dirt on the coverslip then replace with background
        # if (args.process_psf):	
        #     psf = np.where(psf < np.mean(noisy_region), background, psf)
    
        # psf = psf_temp
    
        psf = psf / np.sum(psf)
    
        # Load data PSF onto GPU
        psfs[i] = cp.array(psf, dtype=cp.float32)

    return psfs
        

def rlgc_4D(args, full_image, psf, chstr):
    recon_4D = np.zeros(full_image.shape, dtype=np.float32)
    recon_rl_4D = np.zeros(full_image.shape, dtype=np.float32)
    for timepoint in range(full_image.shape[0]):
        recon, recon_rl = rlgc(args, full_image[timepoint], psf, timepoint, chstr)
        recon_4D[timepoint]= recon
        recon_rl_4D[timepoint]= recon_rl
    return recon_4D, recon_rl_4D

def rlgc_4D_multichannel(args, full_image, psfs):
    recon_4D = np.zeros(full_image.shape, dtype=np.float32)
    recon_rl_4D = np.zeros(full_image.shape, dtype=np.float32)
    for channel in range(len(psfs)):
        recon, recon_rl = rlgc(args, full_image[:,channel], psfs[channel], 0, channel)
        recon_4D[:,channel]= recon
        recon_rl_4D[:,channel]= recon_rl
    return recon_4D, recon_rl_4D

def rlgc_5D(args, full_image,psfs):
    recon_5D = np.zeros(full_image.shape, dtype=np.float32)
    recon_rl_5D = np.zeros(full_image.shape, dtype=np.float32)
    print(full_image.shape)
    
    for channel in range(len(psfs)):
        recon, recon_rl = rlgc_4D(args, full_image[:,:,channel], psfs[channel], channel)
        recon_5D[:,:,channel]= recon
        recon_rl_5D[:,:,channel]= recon_rl
    return recon_5D, recon_rl_5D
        

def rlgc(args, image, psf, tp, ch):
    print(f"Processing timepoint {str(tp)}, channel {str(ch)}")
    
    # Load data and PSF onto GPU
    # TODO: Specify GPU
    image = cp.array(image, dtype=cp.float32)
    
    
    # Calculate OTF and transpose
    # TODO put this in load psf
    otf = cp.fft.rfftn(psf)
    psfT = cp.flip(psf, (0, 1, 2))
    otfT = cp.fft.rfftn(psfT)
    
    
    
    # Get dimensions of data
    num_z = image.shape[0]
    num_y = image.shape[1]
    num_x = image.shape[2]
    num_pixels = num_z * num_y * num_x
    
    # Calculate Richardson-Lucy iterations
    HTones = fftconv(cp.ones_like(image), otfT)
    recon = cp.ones((num_z, num_y, num_x))
    recon_rl = cp.ones((num_z, num_y, num_x))
    
    if (args.iters_output is not None):
        iters = np.zeros((args.max_iters, num_z, num_y, num_x))
    
    if (args.rl_iters_output is not None):
        rl_iters = np.zeros((args.max_iters, num_z, num_y, num_x))
    
    if (args.updates_output is not None):
        updates = np.zeros((args.max_iters, num_z, num_y, num_x))
    
    num_iters = 0
    for iter in range(args.max_iters):
        start_time = timeit.default_timer()
        print(image.shape)
    
        # Split recorded image into 50:50 images
        # TODO: make this work on the GPU (for some reason, we get repeating blocks with a naive conversion to cupy)
        split1 = rng.binomial(image.get().astype('int64'), p=0.5)
        split1 = cp.array(split1)
        split2 = image - split1
    
        # Calculate prediction
        Hu = fftconv(recon, otf)
    
        # Calculate updates for split images and full images (H^T (d / Hu))
        ratio1 = split1 / (0.5 * (Hu + 1E-12))
        ratio2 = split2 / (0.5 * (Hu + 1E-12))
        HTratio1 = fftconv(ratio1, otfT)
        HTratio2 = fftconv(ratio2, otfT)
        ratio = image / (Hu + 1E-12)
        HTratio = fftconv(ratio, otfT)
        HTratio = HTratio / HTones
    
        # Normalise update steps by H^T(1) and only update pixels in full estimate where split updates agree in 'sign'
        update1 = HTratio1 / HTones
        update2 = HTratio2 / HTones
        if (args.blur_consensus != 0):
            shouldNotUpdate = fftconv(fftconv((update1 - 1) * (update2 - 1), otf), otfT) < 0
        else:
            shouldNotUpdate = (update1 - 1) * (update2 - 1) < 0
        HTratio[shouldNotUpdate] = 1
    
        # Save previous estimate to check we're not wasting our time updating small values
        previous_recon = recon
    
        # Update estimate
        recon = recon * HTratio
    
        # Add to full iterations output if asked to by user
        if (args.iters_output is not None):
            iters[iter, :, :, :] = recon.get()
        
        if (args.updates_output is not None):
            updates[iter, :, :, :] = HTratio.get()
    
        # Also calculate normal RL update if asked to by user
        if args.rl_output is not None:
            Hu_rl = fftconv(recon_rl, otf)
            ratio_rl = image / (Hu_rl + 1E-12)
            HTratio_rl = fftconv(ratio_rl, otfT)
            recon_rl = recon_rl * HTratio_rl / HTones
            if (args.rl_iters_output is not None):
                rl_iters[iter, :, :, :] = recon_rl.get()
    
        calc_time = timeit.default_timer() - start_time
        num_updated = num_pixels - cp.sum(shouldNotUpdate)
        max_relative_delta = cp.max((recon - previous_recon) / cp.max(recon))
        print("Iteration %03d completed in %1.3f s. %1.2f %% of image updated. Update range: %1.2f to %1.2f. Largest relative delta = %1.3f" % (iter + 1, calc_time, 100 * num_updated / num_pixels, cp.min(HTratio), cp.max(HTratio), max_relative_delta))
    
        num_iters = num_iters + 1
    
        if (num_updated / num_pixels < args.limit):
            break
    
        if (max_relative_delta < 0.01):
            break
    
    # not implemented for 5D
    # Reblur, collect from GPU and save if argument given
    # if args.reblurred is not None:
    # 	reblurred = fftconv(recon, otf)
    # 	reblurred = reblurred.get()
    # 	tifffile.imwrite(args.reblurred, reblurred, bigtiff=True)
    
    # Collect reconstruction from GPU and save, for now leave on GPU
    # recon = recon.get()
    if args.rl_output is not None:
        # recon_rl = recon_rl.get()
        return recon.get(), recon_rl.get()
    return recon.get()

def fftconv(x, H):
	return cp.fft.irfftn(cp.fft.rfftn(x) * H, x.shape)


if __name__ == '__main__':
	main()
