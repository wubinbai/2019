# Wubin

# The library ImageHash has provided all the functions to implement diff. sorts of hashes: https://pypi.org/project/ImageHash/

# It's an image hashing library written in Python. ImageHash supports:

#    average hashing (aHash)
#    perception hashing (pHash)
#    difference hashing (dHash)
#    wavelet hashing (wHash)



"""
Rationale

Why can we not use md5, sha-1, etc.?

Unfortunately, we cannot use cryptographic hashing algorithms in our implementation. Due to the nature of cryptographic hashing algorithms, very tiny changes in the input file will result in a substantially different hash. In the case of image fingerprinting, we actually want our similar inputs to have similar output hashes as well.
Requirements

Based on PIL/Pillow Image, numpy and scipy.fftpack (for pHash) Easy installation through pypi.
Basic usage

>>> from PIL import Image
>>> import imagehash
>>> hash = imagehash.average_hash(Image.open('test.png'))
>>> print(hash)
d879f8f89b1bbf
>>> otherhash = imagehash.average_hash(Image.open('other.bmp'))
>>> print(otherhash)
ffff3720200ffff
>>> print(hash == otherhash)
False
>>> print(hash - otherhash)
36

The demo script find_similar_images illustrates how to find similar images in a directory.

Source hosted at github: https://github.com/JohannesBuchner/imagehash
Changelog

    4.0: Changed binary to hex implementation, because the previous one was broken for various hash sizes. This change breaks compatibility to previously stored hashes; to convert them from the old encoding, use the “old_hex_to_hash” function.
    3.5: image data handling speed-up
    3.2: whash now also handles smaller-than-hash images

    3.0: dhash had a bug: It computed pixel differences vertically, not horizontally.
        I modified it to follow dHash. The old function is available as dhash_vertical.

    2.0: added whash
    1.0: initial ahash, dhash, phash implementations.

"""

from PIL import Image
import imagehash
hash1a = imagehash.average_hash(Image.open('../images/999900.jpg'))
hash1d = imagehash.dhash(Image.open('../images/999900.jpg'))
hash1p = imagehash.phash(Image.open('../images/999900.jpg'))
print(hash1a)
print(hash1d)
print(hash1p)
hash2a = imagehash.average_hash(Image.open('../images/9999400.jpg'))
hash2d = imagehash.dhash(Image.open('../images/9999400.jpg'))
hash2p = imagehash.phash(Image.open('../images/9999400.jpg'))
print(hash2a)
print(hash2d)
print(hash2p)

print(hash1a-hash2a,hash1d-hash2d,hash1p-hash2p)





### 3

hash3a = imagehash.average_hash(Image.open('../images/9999300.jpg'))
hash3d = imagehash.dhash(Image.open('../images/9999300.jpg'))
hash3p = imagehash.phash(Image.open('../images/9999300.jpg'))
print(hash3a)
print(hash3d)
print(hash3p)

print(hash1a-hash3a,hash1d-hash3d,hash1p-hash3p)

