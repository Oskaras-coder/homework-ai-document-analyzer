import hashlib

"""
    Compute an MD5-based fingerprint of the uploaded file’s contents,
    so identical PDFs always map to the same cache key.
"""
def get_file_hash(file):
    file.seek(0)  # Start reading from the very beginning of the file
    content = file.read()
    file.seek(0)
    return hashlib.md5(content).hexdigest()  # Return the MD5 hash
