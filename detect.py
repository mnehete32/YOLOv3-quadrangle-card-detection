import argparse
import time
import numpy as np
np.set_printoptions(suppress=True)
from models import *
from utils.datasets import *
from utils.utils import *
import pandas as pd
from copy import copy


cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
#device = torch.device('cpu')
parser = argparse.ArgumentParser()

parser.add_argument('-image_folder', type=str, default='trainable_dataset_files/test.part', help='path to images')
parser.add_argument('-output_folder', type=str, default='output_without_rotation/', help='path to outputs')
parser.add_argument('-plot_flag', type=bool, default=True)
parser.add_argument('-txt_out', type=bool, default=True)

parser.add_argument('-cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
parser.add_argument('-weights_path', type=str, default='weights/latest.pt', help='weight file path')
parser.add_argument('-class_path', type=str, default='data/icdar.names', help='path to class label file')
parser.add_argument('-conf_thres', type=float, default=0.1, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.0, help='iou threshold for non-maximum suppression')
# parser.add_argument('-batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('-img_size', type=int, default=32 * 19, help='size of each image dimension')
opt = parser.parse_args()
print(opt)

area_threshold = 0.001
out_fields =  {
    "file_name":"",
    "num_card" : "",
    "P1_x": "",
    "P1_y": "",
    "P2_x": "",
    "P2_y": "",
    "P3_x": "",
    "P3_y": "",
    "P4_x": "",
    "P4_y": "",
    "cls_pred": "",
    "conf": "",
    "cls_conf": "",
}
txt_out_filename = "output_fields.csv"
# if opt.txt_out:
#     df = pd.DataFrame(columns=out_fields.keys())
#     df.to_csv(txt_out_filename,mode="w",index = False)

def get_distances(points):
    card_width = max(np.linalg.norm(points[0] - points[1]),np.linalg.norm(points[3] - points[2]))
    card_height = max(np.linalg.norm(points[1] - points[2]),np.linalg.norm(points[0] - points[3]))
    
    return card_width,card_height



def camscanner(points,shape_wh,img):
    card_width,card_height = shape_wh
    pts1 = np.float32(points[[0,1,3]])
    pts2 = np.float32([[0,0],[card_width,0],[0,card_height]])#[card_width,card_height]])

    # pts1 = np.float32([points[2],points[3],points[0],points[1]])

    matrix = cv2.getAffineTransform(pts1,pts2)
    result = cv2.warpAffine(img, matrix, (int(card_width), int(card_height)))
    return result


def detect(opt):
	# os.system('rm -rf ' + opt.output_folder)
	os.makedirs(opt.output_folder, exist_ok=True)

	# Load model
	model = Darknet(opt.cfg, opt.img_size)

	weights_path = opt.weights_path
	if weights_path.endswith('.weights'):  # saved in darknet format
		load_weights(model, weights_path)
	else:  # endswith('.pt'), saved in pytorch format
		checkpoint = torch.load(weights_path, map_location='cpu')
		model.load_state_dict(checkpoint['model'])
		del checkpoint

	model.to(device).eval()

	# Set Dataloader
	classes = load_classes(opt.class_path)  # Extracts class labels from file
	dataloader = load_images_test(opt.image_folder, batch_size=1, img_size=opt.img_size)
	print(classes)

	# Bounding-box colors
	color_list = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]


	#   imgs = []  # Stores image paths
	#   img_detections = []  # Stores detections for each image index
	prev_time = time.time()

	for batch_i, (img_paths, img,area) in enumerate(dataloader):
		print(batch_i, img.shape)
		path = img_paths[0]
		results_img_path = os.path.join(opt.output_folder, path.split('/')[-1])
		if os.path.isfile(results_img_path.replace('.bmp', '.jpg').replace('.tif', '.jpg')):
			continue
		# Get detections
		with torch.no_grad():
			chip = torch.from_numpy(img).unsqueeze(0).to(device)
			pred = model(chip)
			pred = pred[pred[:, :, 8] > opt.conf_thres]
			detections = [None]
			if len(pred) > 0:
				detections = non_max_suppression_test(pred.unsqueeze(0),area,area_threshold, opt.conf_thres,opt.nms_thres)
	#               img_detections.extend(detections)
	#               imgs.extend(img_paths)

		print('Batch %d... (Done %.3f s)' % (batch_i, time.time() - prev_time))
		prev_time = time.time()

		if len(detections) <1 :
			out_fields_copy = copy(out_fields)
			out_fields_copy["file_name"] = path
			out_fields_copy["num_card"] = 0
			out_row = pd.DataFrame(out_fields_copy,index = [0])
			out_row.to_csv(txt_out_filename, mode = "a", index = False, header = False)
			continue

		if len(detections) == 1 and detections[0] is None:
			continue

		# Iterate through images and save plot of detections
		# for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
		path = img_paths[0]
		detections = detections[0]
	#       print(detections)
		print("image %g: '%s'" % (batch_i,path))
		cropped_ploted = False
		results_img_path = os.path.join(opt.output_folder, path.split('/')[-1])
		cropped_img_path = os.path.join(opt.output_folder, "cropped_"+path.split('/')[-1])

		img = np.uint8(img.transpose(1,2,0) * 255)
		img = np.ascontiguousarray(img, dtype=np.float32)

		# The amount of padding that was added
		pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
		pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
		# Image height and width after padding is removed
		unpad_h = opt.img_size - pad_y
		unpad_w = opt.img_size - pad_x

		# Draw bounding boxes and labels of detections
		if detections is not None:
			unique_classes = detections[:, -1].cpu().unique()
			bbox_colors = random.sample(color_list, len(unique_classes))

			results_txt_path = results_img_path + '.txt'
			if os.path.isfile(results_txt_path):
				os.remove(results_txt_path)

			for i in unique_classes:
				n = (detections[:, -1].cpu() == i).sum()
				print('%g %ss' % (n, classes[int(i)]))

			for num_card, (P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y, conf, cls_conf, cls_pred) in enumerate(detections):
				P1_y = max((((P1_y - pad_y // 2) / unpad_h) * img.shape[0]).round().item(), 0)
				P1_x = max((((P1_x - pad_x // 2) / unpad_w) * img.shape[1]).round().item(), 0)
				P2_y = max((((P2_y - pad_y // 2) / unpad_h) * img.shape[0]).round().item(), 0)
				P2_x = max((((P2_x - pad_x // 2) / unpad_w) * img.shape[1]).round().item(), 0)
				P3_y = max((((P3_y - pad_y // 2) / unpad_h) * img.shape[0]).round().item(), 0)
				P3_x = max((((P3_x - pad_x // 2) / unpad_w) * img.shape[1]).round().item(), 0)
				P4_y = max((((P4_y - pad_y // 2) / unpad_h) * img.shape[0]).round().item(), 0)
				P4_x = max((((P4_x - pad_x // 2) / unpad_w) * img.shape[1]).round().item(), 0)
				
				# write to file
				if opt.txt_out:
					with open(results_txt_path, 'a') as file:
						file.write(('%g %g %g %g %g %g %g %g %g %g \n') % (P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y, cls_pred, cls_conf * conf))
						
					out_fields_copy = copy(out_fields)
					out_fields_copy["file_name"] = path
					out_fields_copy["num_card"] = num_card+1
					out_fields_copy["P1_x"] = P1_x
					out_fields_copy["P1_y"] = P1_y
					out_fields_copy["P2_x"] = P2_x
					out_fields_copy["P2_y"] = P2_y
					out_fields_copy["P3_x"] = P3_x
					out_fields_copy["P3_y"] = P3_y
					out_fields_copy["P4_x"] = P4_x
					out_fields_copy["P4_y"] = P4_y
					out_fields_copy["cls_pred"] = cls_pred.int().item()
					out_fields_copy["conf"] = conf.float().item()
					out_fields_copy["cls_conf"] = cls_conf.float().item()
					out_row = pd.DataFrame(out_fields_copy,index = [0])
					out_row.to_csv(txt_out_filename, mode = "a", index = False, header = False)
					
					

				if opt.plot_flag:
					# Add the bbox to the plot
					label = '%s %.2f' % (classes[int(cls_pred)], conf)
					color = bbox_colors[int(np.where(unique_classes == int(cls_pred))[0])]
					plot_one_box([P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y], img, label=None, color=color,conf = conf)
					poly_arr = np.array([P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y]).reshape(4,2)
	#                   card_width,card_height = get_distances(poly_arr)
	#                   result = camscanner(poly_arr,(card_width,card_height),img)
	#                   cropped_ploted = True
			# cv2.imshow(path.split('/')[-1], img)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

	#       if opt.plot_flag:
		print("saving_file")
	#       img = np.uint8(img.transpose(1,2,0))
		# Save generated image with detections

		print(img.shape,results_img_path.replace('.bmp', '.jpg').replace('.tif', '.jpg'),"\n",
				"IS FILE SAVED:- ",cv2.imwrite(results_img_path.replace('.bmp', '.jpg').replace('.tif', '.jpg'), img[:,:,::-1]))
	#       if cropped_ploted:
	#           print(cv2.imwrite(cropped_img_path.replace('.bmp', '.jpg').replace('.tif', '.jpg'), result))

	torch.cuda.empty_cache()
	detect(opt)

if __name__ == '__main__':
	torch.cuda.empty_cache()
	detect(opt)