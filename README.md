# progressive_nns
This project builds a progressive network as described in the paper 'Progressive Neural Networks' https://arxiv.org/pdf/1606.04671.pdf, and based on the VGG-16 architecture trained for face recognition:

http://www.robots.ox.ac.uk/~albanie/pytorch-models.html

We fine tune this to our own face database and we added a second task that is the one of emotion recognition. For this we took the FER+ database.

The databases we used were msra cfw https://www.microsoft.com/en-us/research/project/msra-cfw-data-set-of-celebrity-faces-on-the-web/ for face recognition and Fer+ can be found here with instructions to download and use https://github.com/Microsoft/FERPlus

To use our code, you need to call progressive_net.py and add arguments, so you should write

python progressive_net.py 'face_path' 'fer_path' 'weights_path' --faces_lr 0.001 --fer_lr 0.0015 --faces_step 10 --fer_step 10 --faces_gamma 0.1 --fer_gamma 0.1 --num_epochs 15
