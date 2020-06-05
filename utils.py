# Read file line-by-line and store in list
def read_file(fh):
    """Read file line-by-line and store in list

    Args:
        fh (string): name of the file

    Returns:
        list: contents of the file
    """
    with open(fh, 'r') as f:
        lines = [line.rstrip() for line in f]
    print(f'Number of Conversations is: {len(lines)}')
    return lines
