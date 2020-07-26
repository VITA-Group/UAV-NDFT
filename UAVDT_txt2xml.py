
# coding: utf-8

# In[1]:


import cv2
import os
cls2name = {1: 'car', 2: 'truck', 3:'bus'}
view2name = {1: 'no-out', 2: 'medium-out', 3: 'small-out'}
occ2name = {1: 'no-occ', 2: 'large-occ', 3: 'medium-occ', 4: 'small-occ'}

# In[3]:


i = 0
gt_path = 'UAVDT/GT'
gts = [gt for gt in sorted(os.listdir(gt_path)) if 'gt_whole' in gt]
vid_anno_dict = {}
for gt in gts:
    vid_name = gt.split('_')[0]
    vid_anno_dict[vid_name] = {}
    print(vid_name)
    with open(os.path.join(gt_path, gt), 'r') as f:
        for line in f:
            i+=1
            line = line.split(',')
            frame_index = int(line[0])
            if frame_index not in vid_anno_dict[vid_name].keys():
                vid_anno_dict[vid_name][frame_index] = {'bboxes':[], 'bboxes_ignore':[]}
            width = int(line[4])
            height = int(line[5])
            x1, y1 = int(line[2])+1, int(line[3])+1
            x2, y2 = x1+width, y1+height
            x2 = 1024 if x2 > 1024 else x2
            y2 = 540 if y2 > 540 else y2
            out_of_view = int(line[6])
            occ = int(line[7])
            cls = int(line[8])
            vid_anno_dict[vid_name][frame_index]['bboxes'].append((x1,y1,x2,y2,out_of_view,occ,cls))
print(i)


# In[4]:


j=0
gt_ignores = [gt_ignore for gt_ignore in sorted(os.listdir(gt_path)) if 'gt_ignore' in gt_ignore]
for gt_ignore in gt_ignores:
    vid_name = gt_ignore.split('_')[0]
    print(vid_name)
    with open(os.path.join(gt_path, gt_ignore), 'r') as f:
        for line in f:
            j += 1
            line = line.split(',')
            frame_index = int(line[0])
            width = int(line[4])
            height = int(line[5])
            x1, y1 = int(line[2]), int(line[3])
            x2, y2 = x1+width, y1+height
            vid_anno_dict[vid_name][frame_index]['bboxes_ignore'].append((x1,y1,x2,y2))
print(j)


# In[6]:


attrs = [os.path.join('M_attr/train', attr) for attr in sorted(os.listdir('M_attr/train'))]             + [os.path.join('M_attr/test', attr) for attr in sorted(os.listdir('M_attr/test'))]
weather_dict = {4:'daylight', 2:'night', 1:'fog'}
altitude_dict = {4:'low-alt', 2:'medium-alt', 1:'high-alt'}
angle_dict = {6:'front-side-view', 4:'front-view', 2:'side-view', 1:'bird-view'}
term_dict = {1:'long-term', 0:'short-term'}
for attr in attrs:
    with open(attr, 'r') as f:
        vid_name = attr.split('/')[-1].split('_')[0]
        attr = f.readlines()[0].strip().split(',')
        #print(attr)
        weather = weather_dict[int(''.join(attr[0:3]), 2)]
        altitude = altitude_dict[int(''.join(attr[3:6]), 2)]
        angle = angle_dict[int(''.join(attr[6:9]), 2)]
        term = term_dict[int(attr[9], 2)]
        #print(weather, altitude, angle, term)
        vid_anno_dict[vid_name]['weather'] = weather
        vid_anno_dict[vid_name]['altitude'] = altitude
        vid_anno_dict[vid_name]['angle'] = angle
        vid_anno_dict[vid_name]['term'] = term



test_vids = ['M0203','M0205','M0208','M0209','M0403','M0601','M0602','M0606','M0701','M0801',
             'M0802','M1001','M1004','M1007','M1009','M1101','M1301','M1302','M1303','M1401']

img_path = 'UAV-benchmark-M'
vids = sorted(os.listdir(img_path))

train_f = open('trainval.txt', 'w')
test_f = open('test.txt', 'w')
for vid_name in vids:
    frames = sorted(os.listdir(os.path.join(img_path, vid_name)))
    for frame in frames:
        frame_index = int(frame[3:9])
        if frame_index not in vid_anno_dict[vid_name].keys():
            continue
        if vid_name not in test_vids:
            train_f.write('{}_{}\n'.format(vid_name, frame_index))
        else:
            test_f.write('{}_{}\n'.format(vid_name, frame_index))


# In[ ]:


from PIL import Image

img_path = 'UAV-benchmark-M'
anno_dir = 'Annotations'
vids = sorted(os.listdir(img_path))
for vid_name in vids:
    weather = vid_anno_dict[vid_name]['weather']
    altitude = vid_anno_dict[vid_name]['altitude']
    angle = vid_anno_dict[vid_name]['angle']
    term = vid_anno_dict[vid_name]['term']
    attributes = {'weather': weather, 'altitude': altitude, 'angle': angle,'term': term}
    frames = sorted(os.listdir(os.path.join(img_path, vid_name)))
    for frame in frames:
        frame_index = int(frame[3:9])
        if frame_index not in vid_anno_dict[vid_name].keys():
            continue
        xml_name = '{}_{}.xml'.format(vid_name, frame_index)
        im = cv2.imread(os.path.join(os.path.join(img_path, vid_name), frame))
        with open(os.path.join(anno_dir, xml_name), 'w') as fout:
            fout.write('<annotation>'+'\n')
            fout.write('\t'+'<folder>VOC2007</folder>'+'\n')
            fout.write('\t'+'<filename>'+'{}_{}.jpg'.format(vid_name, frame_index)+'</filename>'+'\n')

            fout.write('\t'+'<source>'+'\n')
            fout.write('\t\t'+'<database>'+'The UAV_car Database'+'</database>'+'\n')
            fout.write('\t\t'+'<annotation>'+'UAV_car_2017'+'</annotation>'+'\n')
            fout.write('\t\t'+'<image>'+'vatic'+'</image>'+'\n')
            fout.write('\t\t'+'<vaticid>'+'0'+'</vaticid>'+'\n')
            fout.write('\t'+'</source>'+'\n')

            fout.write('\t'+'<owner>'+'\n')
            fout.write('\t\t'+'<flickrid>'+'RandomEvent101'+'</flickrid>'+'\n')
            fout.write('\t\t'+'<name>'+'?'+'</name>'+'\n')
            fout.write('\t'+'</owner>'+'\n')

            fout.write('\t'+'<size>'+'\n')
            fout.write('\t\t'+'<width>'+str(im.shape[0])+'</width>'+'\n')
            fout.write('\t\t'+'<height>'+str(im.shape[1])+'</height>'+'\n')
            fout.write('\t\t'+'<depth>'+'3'+'</depth>'+'\n')
            fout.write('\t'+'</size>'+'\n')

            fout.write('\t'+'<segmented>'+'0'+'</segmented>'+'\n')
            fout.write('\t'+'<weather>'+weather+'</weather>'+'\n')
            fout.write('\t'+'<altitude>'+altitude+'</altitude>'+'\n')
            fout.write('\t'+'<angle>'+angle+'</angle>'+'\n')
            fout.write('\t'+'<term>'+term+'</term>'+'\n')

            for bbox in vid_anno_dict[vid_name][frame_index]['bboxes']:
                fout.write('\t'+'<object>'+'\n')
                #fout.write('\t\t'+'<name>'+cls2name[bbox[6]]+'</name>'+'\n')
                fout.write('\t\t'+'<name>'+'car'+'</name>'+'\n')
                fout.write('\t\t'+'<pose>'+'Unspecified'+'</pose>'+'\n')
                fout.write('\t\t'+'<difficult>'+'0'+'</difficult>'+'\n')
                fout.write('\t\t'+'<out-of-view>'+view2name[bbox[4]]+'</out-of-view>'+'\n')
                fout.write('\t\t'+'<occlusion>'+occ2name[bbox[5]]+'</occlusion>'+'\n')
                fout.write('\t\t'+'<bndbox>'+'\n')
                fout.write('\t\t\t'+'<xmin>'+str(bbox[0])+'</xmin>'+'\n')
                fout.write('\t\t\t'+'<ymin>'+str(bbox[1])+'</ymin>'+'\n')
                # pay attention to this point!(0-based)
                fout.write('\t\t\t'+'<xmax>'+str(bbox[2])+'</xmax>'+'\n')
                fout.write('\t\t\t'+'<ymax>'+str(bbox[3])+'</ymax>'+'\n')
                fout.write('\t\t'+'</bndbox>'+'\n')
                fout.write('\t'+'</object>'+'\n')

            fout.write('</annotation>')
