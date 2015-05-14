
import sys, struct
print sys.argv[1]
with open(sys.argv[1], "rb") as f:
	for line in f:
		for c in line:
			print(c)


# This example demonstrates how to read a binary file, by reading the width and
# height information from a bitmap file. First, the bytes are read, and then
# they are converted to integers.

# When reading a binary file, always add a 'b' to the file open mode
with open(sys.argv[1], 'rb') as f:
    # BMP files store their width and height statring at byte 18 (12h), so seek
    # to that position
    f.seek(18)

    # The width and height are 4 bytes each, so read 8 bytes to get both of them
    bytes = f.read(8)

    # Here, we decode the byte array from the last step. The width and height
    # are each unsigned, little endian, 4 byte integers, so they have the format
    # code '<II'. See http://docs.python.org/3/library/struct.html for more info
    size = struct.unpack('<II', bytes)

    # Print the width and height of the image
    print('Image width:  ' + str(size[0]))
    print('Image height: ' + str(size[1]))
