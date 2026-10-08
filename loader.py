with open(filepath, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

header_len = struct.unpack("<Q", mm[:8])[0]