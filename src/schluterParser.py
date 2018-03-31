import csv

def annotationCvParser(annotation_filename):
    """
    Schluter onset time annotation parser
    :param annotation_filename:
    :return: onset time list
    """
    list_onset_time = []
    with open(annotation_filename, 'rb') as file:
        annotation = csv.reader(file)
        for onset_time in annotation:
            list_onset_time.append(onset_time[0])
    return list_onset_time


if __name__ == '__main__':
    from file_path_bock import *

    test_annotation_filename = join(bock_annotations_path, 'ah_development_guitar_2684_TexasMusicForge_Dandelion_pt1.onsets')
    list_onset_time = annotationCvParser(test_annotation_filename)
    print(list_onset_time)

    test_cv_filename = join(bock_cv_path, '8-fold_cv_random_0.fold')
    list_test_filename = annotationCvParser(test_cv_filename)
    print(list_test_filename)
