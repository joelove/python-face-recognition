def build_path(*parts):
    return '/'.join(parts)

def name_from_path(path):
    filename = path.split('/')[-1:]
    name = filename.split('.')[0]
    return name
