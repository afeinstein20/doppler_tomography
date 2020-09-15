import getopt
import os, sys
import numpy as np
from tqdm import tqdm
from astropy.io import fits
import matplotlib.pyplot as plt


def list_files(dir):
    """
    Creates an array of files in the given directory. Looking specifically for FITS files.
    Also defines the number of unique orders in a file and the length of each order. This 
         is necessary because the files were reduced irregularly.
    """
    files = np.sort([os.path.join(dir, i) for i in os.listdir(dir) if i.endswith('.fits')])
    
    if len(files) == 0:
        print('No FITS files in this directory.')
        sys.exit()

    else:
        hdu = fits.open(files[0])

        unique_orders = np.unique(hdu[0].data[0])
        length_per = np.zeros(len(unique_orders), dtype=int)
    
        for j in range(len(unique_orders)):
            length_per[j] = len(hdu[0].data[4][hdu[0].data[0]==unique_orders[j]])
        hdu.close()
    
        return files, unique_orders, length_per


def load_data(files, unique_orders, length_per):
    """
    Loads wavelength, spectra, errors, and orders into 2D arrays.
    """
    WAVES   = np.zeros((len(files), np.nansum(length_per)))
    SPECTRA = np.zeros((len(files), np.nansum(length_per)))
    ERRS    = np.zeros((len(files), np.nansum(length_per)))
    ORDERS  = np.zeros((len(files), np.nansum(length_per)))

    for i in tqdm(range(len(files))):
        hdu = fits.open(files[i])

        for j in range(len(unique_orders)):
            o = hdu[0].data[0]
            q = o == unique_orders[j]
            w = hdu[0].data[4][q]
            s = hdu[0].data[10][q]
            e = hdu[0].data[11][q]

            reg = [np.nansum(length_per[:j]), np.nansum(length_per[:j])+length_per[j]]

            WAVES[i][reg[0]:reg[1]] = w
            SPECTRA[i][reg[0]:reg[1]] = s
            ERRS[i][reg[0]:reg[1]] = e
            ORDERS[i][reg[0]:reg[1]] = o[q]

        hdu.close()

    return WAVES, SPECTRA, ERRS, ORDERS

def main(argv):
    """
    Arg parser :o
    """
    directory = ''
    outputfile = ''
    interpolate = False

    try:
        opts, args = getopt.getopt(argv, 'hd:o:', ['directory=', 'ofile=', 'interpolate='])
    except getopt.GetoptError:
        print('extract.py -d <data_path> -o <output_file> -i <interpolate_bool>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('extract.py -d <data_path> -o <output_file> -i <interpolate_bool>')
            sys.exit()
    
        elif opt in ('-d', '--data_path'):
            directory = arg

        elif opt in ('-o', '--output_file'):
            outputfile = arg

        elif opt in ('-i', '--interpolate'):
            interpolate = arg

    return directory, outputfile, interpolate


if __name__ == "__main__":
    dir, ofile, interp = main(sys.argv[1:])
    files, unique_orders, length_per = list_files(dir)
    w, s, e, o = load_data(files, unique_orders, length_per)

    if interp:
        print('Havent integrated this yet')

    if ofile is not '':
        np.save(os.path.join(dir, ofile), np.array([w, s, e, o]))
    else:
        return w, s, e, o
    
