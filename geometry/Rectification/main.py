from skimage import io

from rectification import rectification

if __name__ == '__main__':
    image_name = "name_of_image"
    image_name = "geometry/Rectification/3437-7029.jpg"
    image = io.imread(image_name)
    print("Rectifying {}".format(image_name))
    save_name = '.'.join(image_name.split('.')[:-1]) + '_warped.png'
    rect = rectification(image_name)
    io.imsave(save_name, rect.rectify_image(clip_factor=4, algorithm='independent'))
