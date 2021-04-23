import numpy as np  
import sys,os  
import cv2
import xml.etree.ElementTree as ET
import pdb
from caffe.proto import caffe_pb2
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
caffe_root = '/opt/mobilenetssd/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe
import time
  
save_dir = '/opt/mobilenetssd/caffe/examples/MobileNet-SSD/xml/'

# net_file= 'no_bn_ccfa.prototxt'  
# caffe_model='no_bn_ccfa.caffemodel'  
net_file= 'no_bn_0406_paimian.prototxt'  
caffe_model='no_bn_0406_paimian.caffemodel'  
#test_dir = "testSet"

# small_img_w_length = 1000
# small_img_h_length = 1000
# small_img_hw_length = 1000
# tmp = small_img_hw_length - 30

img_overlap = 60
image_path = "testSet"
small_img_path = "small_image"
small_image_result = "small_image_result"

if not os.path.exists(caffe_model):
    print(caffe_model + " does not exist")
    exit()
if not os.path.exists(net_file):
    print(net_file + " does not exist")
    exit()
    
net = caffe.Net(net_file,caffe_model,caffe.TEST)  


def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img


def get_groups(len_of_img, overlap, small_img_length):
    length = small_img_length - overlap

    if len_of_img % length < overlap:
        last_one = 0
    else:
        last_one = 1

    return int(len_of_img / length) + last_one


def split_image(image_file, small_img_hw_length):
    img = cv2.imread(os.path.join(image_path,image_file))
    img_height, img_width, _ = img.shape
    print("img_height: {}".format(img_height))
    print("img_width: {}".format(img_width))

    y_groups = get_groups(img_height, img_overlap, small_img_hw_length )
    x_groups = get_groups(img_width, img_overlap, small_img_hw_length )

    sImageInfo = []
    for x in range(x_groups):
        for y in range(y_groups):
            if x == x_groups or y == y_groups: continue

            y_start = y * (small_img_hw_length - img_overlap)
            
            if y == y_groups - 1:
                y_end = img_height
            else:
                y_end = y_start + small_img_hw_length
            
            x_start = x * (small_img_hw_length - img_overlap)
            
            if x == x_groups - 1:
                x_end = img_width
            else:
                x_end = x_start + small_img_hw_length
            
            small_img = img[y_start:y_end, x_start:x_end]
            img_file_name = '{}_{}_{}'.format(x,y,image_file)

            ih = y_end-y_start
            iw = x_end-x_start
            #img_file_name = img_file_name.split("/")[1]
            if ih != iw :
                eh =  max(ih, iw)
                scale = min(eh * 1.0/ ih, eh * 1.0/ iw)

                nh = int(ih * scale)
                nw = int(iw * scale)
                #image = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
                top = (eh - ih) // 2
                print("top:",top)
                bottom = eh - ih - top
                left = (eh - iw) // 2
                print("left",left)
                right = eh - iw - left
                shrink = cv2.copyMakeBorder(small_img, top, bottom, left, right, cv2.BORDER_CONSTANT)
                #new_jpg_file = path_new + jpg_file
                #cv2.imwrite(new_jpg_file, shrink)
                cv2.imwrite("small_image/{}".format(img_file_name), shrink)
                sImageInfo.append((x_start, x_end, y_start, y_end, left, top, img_file_name))

            else:
                cv2.imwrite("small_image/{}".format(img_file_name), small_img)
                sImageInfo.append((x_start, x_end, y_start, y_end, 0, 0, img_file_name))

    return sImageInfo

def save_small_result(image_file, result):

    print("image_file:",image_file)
    bbox_num = 0

    img = cv2.imread(os.path.join(small_img_path, image_file))
    height, width, channels = img.shape
    print (width, height)
    for item in result:
        bbox_num = bbox_num + 1

        xmin = int(round(item[0] * width))
        ymin = int(round(item[1] * height))
        xmax = int(round(item[2] * width))
        ymax = int(round(item[3] * height))
   

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)
        confidence = round(item[-2],1)
        # cv2.putText(img, item[-1] + ":" + str(confidence), (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(200,255,155),thickness=2)
        cv2.putText(img, str(confidence), (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(200,255,155),thickness=2)
       # print (item)
        #print ([xmin, ymin, xmax, ymax])
        #print ([xmin, ymin], item[-1])
    print("bbox_num:",bbox_num)
    #cv2.imshow(image_file,img)
    #cv2.waitKey(0)
    #cv2.destroyWindow(image_file)
    cv2.imwrite("small_image_result/{}".format(image_file), img)

def my_nms(dets, thresh):
    
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    order = scores.argsort()[::-1]
    

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

       
      
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def save_result(image_file, dets, pic_len, thresh=0.015):
    
    #savefileName = '{}.txt'.format(image_file)
    #txtfile = open(savefileName, 'w')

    # tmp = pic_len - img_overlap

    img = cv2.imread(os.path.join(image_path,image_file))
    height, width, channels = img.shape
    print(width, height)
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    # #rm tiny
    for i in inds:
        bbox2 = dets[i, :4]
        px = int(bbox2[2]-bbox2[0])
        py = int(bbox2[3]-bbox2[1])
        if px==0:px=1
        if py==0:py=1
        if (py/px>7 and py>px) or (px/py>7 and px>py):
            dets[i,:4] = [0,0,0,0]


    #merge container

    for i in inds:
        bbox1 = dets[i, :4]

        for j in inds:
            bbox2 = dets[j, :4]
            if i == j:continue
            AJoin3 = 0
            if bbox1[2]>bbox2[0] and bbox2[2]>bbox1[0] and bbox1[3]>bbox2[1] and bbox2[3]>bbox1[1]:
                p1x = max(bbox1[0],bbox2[0])
                p1y = max(bbox1[1],bbox2[1])
                p2x = min(bbox1[2],bbox2[2])
                p2y = min(bbox1[3],bbox2[3])            
                if p2x > p1x and p2y > p1y :
                    AJoin3 = (p2x - p1x)*(p2y - p1y) 
                A3 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
                A4 = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
                Amin3 = min(A3, A4)
                Amax3 = max(A3, A4)

                if AJoin3/Amin3>0.5 :
                    #dets[i,:4] = [min(bbox1[0],bbox2[0]), min(bbox1[1],bbox2[1]), max(bbox1[2],bbox2[2]), max(bbox1[2],bbox2[3])]
                    #dets[j,:4] = [0,0,0,0]
                    if Amax3 == A4:
                        #dets[i,:4] = [min(bbox1[0],bbox2[0]), min(bbox1[1],bbox2[1]), max(bbox1[2],bbox2[2]), max(bbox1[2],bbox2[3])]
                        dets[i,:4] = [bbox2[0], bbox2[1], bbox2[2], bbox2[3]]
                        dets[j,:4] = [0,0,0,0]
                    else:
                        dets[i,:4] = [bbox1[0], bbox1[1], bbox1[2], bbox1[3]]
                        dets[j,:4] = [0,0,0,0]

    #merge up and down bbox
    #dets1 = dets
    for i in inds:
        bbox1 = dets[i, :4]
        Join = 0
        index = 0
        #print("-------------i---------:",bbox1[0], bbox1[1], bbox1[2], bbox1[3])
        AJoin1 = 0
        for j in inds:
            bbox2 = dets[j, :4]
            if i == j or (bbox1[0] == bbox2[0] and bbox1[1] == bbox2[1] and bbox1[2] == bbox2[2] and bbox1[3] == bbox2[3]):continue
                    
            if bbox1[2]>bbox2[0] and bbox2[2]>bbox1[0] and bbox1[3]>bbox2[1] and bbox2[3]>bbox1[1]:
                p1x = max(bbox1[0],bbox2[0])
                p1y = max(bbox1[1],bbox2[1])
                p2x = min(bbox1[2],bbox2[2])
                p2y = min(bbox1[3],bbox2[3])
            
                if p2x > p1x and p2y > p1y :#and (p2x - p1x)>(p2y - p1y)
                    AJoin1 = (p2x - p1x)*(p2y - p1y) 
                A1 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
                A2 = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
                Amin1 = min(A1, A2)
                Amax1 = max(A1, A2)

                mid = (p1y + p2y)/2
                pos0 = mid % (pic_len - img_overlap)
                
                if AJoin1>Join and (0 < pos0 < 30):
                #if AJoin1>Join and ((960<p1y and p2y<1010) or (960*2<p1y and p2y<970*2+60) or (960*3<p1y and p2y<970*3+60) or (960*4<p1y and p2y<970*4+60)):
                # if AJoin1>Join and ((970<mid and mid<1000) or (970*2<mid and mid<970*2+30) or (970*3<mid and mid<970*3+30) or (970*4<mid and mid<970*4+30) or(970*5<mid and mid<970*5+30)):
                    Join = AJoin1
                    index = j

        if Join == 0 :continue 
        #print("index:",index)
        
        bbox4 = dets[index, :4]
        p1x = max(bbox1[0],bbox4[0])
        p1y = max(bbox1[1],bbox4[1])
        p2x = min(bbox1[2],bbox4[2])
        p2y = min(bbox1[3],bbox4[3])
        A1 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
        A4 = (bbox4[2]-bbox4[0]) * (bbox4[3]-bbox4[1])
        Amin = min(A1, A4)
        Amax = max(A1, A4)
        if (((p2x - p1x)/(p2y - p1y)>1.5 or (p2x - p1x)/min(bbox1[2]-bbox1[0], bbox4[2]-bbox4[0])>0.35) and (p2y - p1y)>10) or (p2y - p1y >= 30):
            #dets[i,:4] = [min(bbox1[0],bbox4[0]),min(bbox1[1],bbox4[1]),max(bbox1[2],bbox4[2]),max(bbox1[3],bbox4[3])]
            if Amax/Amin >6:
                dets[i,:4] = [min(bbox1[0], bbox4[0]),min(bbox1[1],bbox4[1]), max(bbox1[2],bbox4[2]),max(bbox1[3],bbox4[3])]
                dets[index,:4] = [0,0,0,0]
                continue
            if (bbox4[3]-bbox4[1])/(max(bbox1[3],bbox4[3])-min(bbox1[1],bbox4[1]))>0.6:
                dets[i,:4] = [bbox4[0],min(bbox1[1],bbox4[1]), bbox4[2],max(bbox1[3],bbox4[3])]
                dets[index,:4] = [0,0,0,0]
                continue
            if (bbox1[3]-bbox1[1])/(max(bbox1[3],bbox4[3])-min(bbox1[1],bbox4[1]))>0.6:
                dets[i,:4] = [bbox1[0],min(bbox1[1],bbox4[1]), bbox1[2],max(bbox1[3],bbox4[3])]
                dets[index,:4] = [0,0,0,0]
                continue

            dets[i,:4] = [(bbox1[0] + bbox4[0])/2,min(bbox1[1],bbox4[1]), (bbox1[2] + bbox4[2])/2,max(bbox1[3],bbox4[3])]
            # dets[i,:4] = [min(bbox1[0], bbox4[0]),min(bbox1[1],bbox4[1]), max(bbox1[2],bbox4[2]),max(bbox1[3],bbox4[3])]
            dets[index,:4] = [0,0,0,0]
    
    #merge right and left bbox
    for i in inds:
        bbox1 = dets[i, :4]
        AJoin = 0
        Amin = 0
        index = 0
        AJoin1 = 0
        for j in inds:
            bbox2 = dets[j, :4]
            if i == j or (bbox1[0] == bbox2[0] and bbox1[1] == bbox2[1] and bbox1[2] == bbox2[2] and bbox1[3] == bbox2[3]):continue
            #AJoin1 = 0
            if bbox1[2]>bbox2[0] and bbox2[2]>bbox1[0] and bbox1[3]>bbox2[1] and bbox2[3]>bbox1[1]:
                p1x = max(bbox1[0],bbox2[0])
                p1y = max(bbox1[1],bbox2[1])
                p2x = min(bbox1[2],bbox2[2])
                p2y = min(bbox1[3],bbox2[3])            
                if p2x > p1x and p2y > p1y :#and (p2y - p1y)>(p2x - p1x)
                    AJoin1 = (p2x - p1x)*(p2y - p1y) 
                A1 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
                A2 = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
                Amin1 = min(A1, A2)
                Amax = max(A1, A2)
                middle = (p1x+p2x)/2
                pos = middle % (pic_len - img_overlap)
                    # if AJoin1 > AJoin and (0 < pos < 30):
                if AJoin1>AJoin and (0 < pos < 30):
                #if AJoin1>AJoin and ((960<p1x and p2x<1010) or (960*2<p1x and p2x<970*2+60) or (960*3<p1x and p2x<970*3+60) or (960*4<p1x and p2x<970*4+60)):
                # if AJoin1>AJoin and ((970<middle and middle<1000) or (970*2<middle and middle<970*2+30) or (970*3<middle and middle<970*3+30) or (970*4<middle and middle<970*4+30)):
                    AJoin = AJoin1
                    index = j
                    Amin = Amin1        
        #if Amin>0 and AJoin/Amin>0.3:
        if AJoin == 0:continue
        #print("------------yesbefore")
        bbox = dets[index, :4]
        p1x = max(bbox1[0],bbox[0])
        p1y = max(bbox1[1],bbox[1])
        p2x = min(bbox1[2],bbox[2])
        p2y = min(bbox1[3],bbox[3]) 
        if ((p2y - p1y)/(p2x - p1x)>3 and AJoin/Amin>0.1) or AJoin/Amin>0.20 or (p2x - p1x >= 30):
            #dets[i,:4] = [min(bbox1[0],bbox[0]),min(bbox1[1],bbox[1]),max(bbox1[2],bbox[2]),max(bbox1[3],bbox[3])]
            dets[i,:4] = [min(bbox1[0],bbox[0]), min(bbox1[1], bbox[1]), max(bbox1[2],bbox[2]), max(bbox1[3], bbox[3])]
            dets[index,:4] = [0,0,0,0]


    #rm tiny
    for i in inds:
        bbox2 = dets[i, :4]
        px = int(bbox2[2]-bbox2[0])
        py = int(bbox2[3]-bbox2[1])
        if px==0:px=1
        if py==0:py=1
        if (py/px>5 and py>px) or (px/py>5 and px>py):
            dets[i,:4] = [0,0,0,0]

    #nms
    nms_ret = my_nms(dets, 0.25)
    dets = dets[nms_ret, :]
    inds = np.where(dets[:, -1] >= thresh)[0]

    #output bbox xml message
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'IMAGE'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_file

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width
    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % 3
   
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        #rm [0,0,0,0]
        if bbox[0]==0 and bbox[2]==0 and bbox[1]==0 and bbox[3]==0 : continue

        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = 'object'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % bbox[0]
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % bbox[1]
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % bbox[2]
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bbox[3]


        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 5)
        confidence = round(score,1)
        # cv2.putText(img, item[-1] + ":" + str(confidence), (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(200,255,155),thickness=2)
        cv2.putText(img, str(confidence), (bbox[0], bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(200,255,155),thickness=2)

    xml = tostring(node_root, pretty_print=True)    
    dom = parseString(xml)
    if image_file.split(".")[1] == 'png':
        save_ml = os.path.join(save_dir, image_file.replace('png', 'xml'))
    else:
        save_ml = os.path.join(save_dir, image_file.replace('jpg', 'xml'))
    with open(save_ml, 'wb') as f:
        f.write(xml)


    #txtfile.close()

    #image = cv2.resize(img, (width/2, height/2))
    cv2.imwrite("testResult/{}".format(str(image_file)), img) 


def detect(imgfile):
    origimg = cv2.imread(os.path.join(small_img_path, imgfile))
    img = preprocess(origimg)

    #img = preprocess(imgfile)
    
    img = img.astype(np.float32)

    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward() 
    
    conf_thresh = 0.15
    topn = 50 
    h = img.shape[0]
    w = img.shape[1]

    det_label = out['detection_out'][0,0,:,1]
    det_conf = out['detection_out'][0,0,:,2]
    det_xmin = out['detection_out'][0,0,:,3]
    det_ymin = out['detection_out'][0,0,:,4]
    det_xmax = out['detection_out'][0,0,:,5]
    det_ymax = out['detection_out'][0,0,:,6]


    top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    #top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    result = []
    #for i in range(len(box)):
    for i in range(min(topn, top_conf.shape[0])):
        xmin = top_xmin[i] 
        ymin = top_ymin[i] 
        xmax = top_xmax[i] 
        ymax = top_ymax[i] 
        score = top_conf[i]
        label = int(top_label_indices[i])
        #label_name = top_labels[i]
        #result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        result.append([xmin, ymin, xmax, ymax, label, score])
    return result
   # cv2.imwrite("SSD.jpg", origimg)
   # filename=imgfile.split("/")[1]
    #print(filename)

    #cv2.imwrite("testResult/{}".format(str(filename)), origimg)

if __name__=='__main__':
    
    im_names = []
    # pic_size = small_img_hw_length 
    
    for filename in os.listdir(os.path.join(image_path, '')):
        if filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".png"):
            im_names.append(filename)
    print(im_names)

    timeRes = []

    tx = time.time()
    nPic = 0
    for image_file in im_names:
        try:
            nPic += 1
            # Split to small images
            t0 = time.time()
            img = cv2.imread(os.path.join(image_path,image_file))

            t1 = time.time()

            img_height, img_width, _ = img.shape
            
            if img_height<=2500 or img_width<=2500:pic_size = 600
            else:pic_size = 600
            # pic_size = 600

            split_info = split_image(image_file ,pic_size)
            
            t2 = time.time()

            results = np.empty([1, 5], dtype=np.float32)
            print (results.shape )
            #split_info:(x_start, x_end, y_start, y_end, left, top, img_file_name)
            for small_image in split_info:
                result = detect(small_image[6])
                save_small_result(small_image[6], result)
                
                sWidth = small_image[1] - small_image[0]
                sHeight = small_image[3] - small_image[2]

                print("image_file:",small_image[6])
                bbox_num = 0

                print ("sWidth,sHeight:",sWidth,sHeight)
                s = max(sWidth,sHeight)
                print ("s:",s)         
                #result:[xmin, ymin, xmax, ymax, label, score]
                # Merge small images' result to orignal image
                for index, item in enumerate(result):
                    bbox_num = bbox_num + 1
                    result[index][0] = int(round(item[0] * s)) + small_image[0] -small_image[4] 
                    result[index][1] = int(round(item[1] * s)) + small_image[2] - small_image[5]
                    result[index][2] = int(round(item[2] * s)) + small_image[0] -small_image[4]
                    result[index][3] = int(round(item[3] * s)) + small_image[2] - small_image[5]
                    tmp_nd = np.array([[result[index][0],result[index][1],result[index][2],result[index][3],result[index][5]]]).astype(np.float32)
                    results = np.vstack((results, tmp_nd))
                print("bbox_num:",bbox_num)
            #print(results.shape)
            nms_ret = my_nms(results, 10)
            # print(nms_ret)
            final_results = results[nms_ret, :]
            save_result(image_file, final_results, pic_size)
            t3 = time.time()
            print("get pic time: %f  inference time: %f  time of all: %f"%(t1-t0,t3-t1,t3-t0))
            timeRes.append([image_file,t1-t0,t3-t1,t3-t0])
        except:
            pass
    ty = time.time()
    for i in timeRes:
        print("picName:%s get pic time: %f  inference time: %f  time of all: %f"%(i[0],i[1],i[2],i[3]))
    print("%d pic inference times: %f "%(nPic,ty-tx))

