import string


def dmstod(data):
    d = int(data)
    m = int((data - d) * 60)
    s = int(((data - d) * 60 - m) * 60)
    return str(d) + '°' + str(m) + '′' + str(s) + '″'

print(dmstod(122.418612))