from flask import Flask
import os
import model.Augmentator as Aug
app = Flask(__name__)

@app.route('/')
def hello_world():
    aug_test()
    return "Hello, World!"

def aug_test():
    print('aug_test!!!')
    # image_dir = './test_data/train/images/'
    # label_dir = './test_data/train/labels/'
    # output_image_dir = './test_data/train/aug_images/'
    # output_label_dir = './test_data/train/aug_labels/'

    image_dir = './test_data/single_data/train/images/'
    label_dir = './test_data/single_data/train/labels/'
    output_image_dir = './test_data/single_data/train/aug_images/'
    output_label_dir = './test_data/single_data/train/aug_labels/'

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    aug = Aug.Augmentator(
        image_dir=image_dir,
        label_dir=label_dir,
        output_image_dir=output_image_dir,
        output_label_dir=output_label_dir,
        transform=Aug.transform,
        output_num=1, # 한 이미지로 몇장의 증강 데이터를 만들지 결정
    )
    aug.run()
    
    synth = Aug.ObjectSynthesizer(
        image_dir=image_dir,
        label_dir=label_dir,
        output_image_dir=output_image_dir,
        output_label_dir=output_label_dir,
        obj_num=3, 
    )
    synth.run()

if __name__ == '__main__':
    aug_test()
    app.run(debug=True, host='0.0.0.0')
