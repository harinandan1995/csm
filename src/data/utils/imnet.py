
imnet_class2sysnet = {
    'horse': 'n02381460',
    'zebra': 'n02391049',
    'bear': 'n02131653',
    'sheep': 'n10588074',
    'cow': 'n01887787'
}


def get_sysnet_id_for_imnet_class(imnet_class):

    if imnet_class in imnet_class2sysnet:
        return imnet_class2sysnet[imnet_class]
    else:
        raise ValueError('Image net class %s not found' % imnet_class)
