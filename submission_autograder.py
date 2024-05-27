from zipfile import ZipFile
import os, sys

project_name = "P1 - Machine Learning"

submission_filename = "submission-p1.zip"

submitted_files = ['models.py']

if __name__ == '__main__':

    if os.path.exists(submission_filename):
        os.remove(submission_filename)

    abort = False
    for filename in submitted_files:
        if not os.path.isfile(filename):
            print("You must submit file {} and I cannot find it.".format(filename))
            abort = True
    if abort:
        sys.stdout.flush()
        sys.exit(1)
        
    with ZipFile(submission_filename, 'w') as zipObj:
        # Iterate over all the files in directory
        for filename in submitted_files:
            zipObj.write(filename)

    print("Now submit the file {} to project \'{}\' on Gradescope.".format(submission_filename, project_name))
