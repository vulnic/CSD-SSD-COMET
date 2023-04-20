import argparse
import json
import os
from tqdm import tqdm

def main(args):
    voc12_json = json.load(open(args.voc12_json,'r'))
    voc07_json = json.load(open(args.voc07_json,'r'))

    for i in range(len(voc12_json['images'])):
        pth = voc12_json['images'][i]['file_name']
        voc12_json['images'][i]['file_name'] = os.path.join(args.voc12_images,pth)
    for i in range(len(voc07_json['images'])):
        pth = voc07_json['images'][i]['file_name']
        voc07_json['images'][i]['file_name'] = os.path.join(args.voc07_images,pth)

    with open(args.voc12_json,'w') as f:
        json.dump(voc12_json,f)
        
    with open(args.voc07_json,'w') as f:
        json.dump(voc07_json,f)

    # voc07_imid2annots = {x['id']:[] for x in voc07_json['images']}
    # for annot in voc07_json['annotations']:
    #     imid = annot['image_id']
    #     voc07_imid2annots[imid].append(annot)

    # # filtered down to voc07 images
    # voc12_ids_filt = set([int(str(x['id'])[4:]) for x in voc12_json['images'] if str(x['id'])[0:4]=='2007'])

    # # find images not in voc12
    # imgs_not_in_voc12 = []
    # for image in tqdm(voc07_json['images']):
    #     if int(image['id']) not in voc12_ids_filt:
    #         imgs_not_in_voc12.append(image)

    # # find the annotations of those images
    # anns_not_in_voc12 = []
    # for image in tqdm(imgs_not_in_voc12):
    #     anns_not_in_voc12.append(voc07_imid2annots[image['id']])

    # voc0712 = voc12_json.copy()
    # voc0712['images'] += imgs_not_in_voc12
    # voc0712['annotations'] += anns_not_in_voc12

    # with open(args.output,'w') as f:
    #     json.dump(voc0712,f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc12_json",type=str)
    parser.add_argument("--voc07_json",type=str)
    parser.add_argument("--voc12_images",type=str)
    parser.add_argument("--voc07_images",type=str)
    parser.add_argument("--output",type=str)
    args = parser.parse_args()
    main(args)