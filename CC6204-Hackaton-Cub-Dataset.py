"""CC6204-Hackaton-Cub-Dataset: Multimodal"""
import glob
import os
import re
import datasets

from requests import get

#datasets.logging.set_verbosity_debug()
#logger = datasets.logging.get_logger(__name__)
#datasets.logging.set_verbosity_info()
#datasets.logging.set_verbosity_debug()


_DESCRIPTION = "Dataset multimodal para actividad del hackaton curso CC6204: Deep Learning"
_CITATION = "XYZ"
_HOMEPAGE = "https://github.com/ivansipiran/CC6204-Deep-Learning/blob/main/Hackaton/hackaton.md"

_REPO = "https://huggingface.co/datasets/alkzar90/CC6204-Hackaton-Cub-Dataset/resolve/main/data"

_URLS = {
   "train_test_split": f"{_REPO}/train_test_split.txt",
   "classes": f"{_REPO}/classes.txt",
   "image_class_labels": f"{_REPO}/image_class_labels.txt",
   "images": f"{_REPO}/images.txt",
   "image_urls": f"{_REPO}/images.zip",
   "text_urls": f"{_REPO}/text.zip",
   "mini_images_urls": f"{_REPO}/dummy/mini_images.zip"
}

# Create ClassId-to-label dictionary using the classes file
classes = get(_URLS["classes"]).iter_lines()
_ID2LABEL = {}
for row in classes:
   row = row.decode("UTF8")
   if row != "":
      idx, label = row.split(" ")
      _ID2LABEL[int(idx)] = re.search("[^\d\.\_+].+", label).group(0).replace("_", " ")
      
      
_NAMES = list(_ID2LABEL.values())

# Create imageId-to-ClassID dictionary using the image_class_labels
img_idx_2_class_idx = get(_URLS["image_class_labels"]).iter_lines()
_IMGID2CLASSID = {}
for row in img_idx_2_class_idx:
   row = row.decode("UTF8")
   if row != "":
      idx, class_id = row.split(" ")
      _IMGID2CLASSID[idx] = int(class_id)
      

# build from images.txt: a mapping from image_file_name -> id
imgpath_to_ids = get(_URLS["images"]).iter_lines()
_IMGNAME2ID = {}
for row in imgpath_to_ids:
   row = row.decode("UTF8")
   if row != "":
      idx, img_name = row.split(" ")
      _IMGNAME2ID[os.path.basename(img_name)] = idx
  
    
# Create TRAIN_IDX_SET
train_test_split = get(_URLS["train_test_split"]).iter_lines()
_TRAIN_IDX_SET = []
for row in train_test_split:
   row = row.decode("UTF8")
   if row != "":
      idx, train_bool = row.split(" ")
      # 1: train, 0: test
      if train_bool == "1":
         _TRAIN_IDX_SET.append(idx)
         
_TRAIN_IDX_SET = set(_TRAIN_IDX_SET)


class CubDataset(datasets.GeneratorBasedBuilder):
   """Cub Dataset para el Hackaton del curso CC6204: Deep Learning"""
   
   def _info(self):
      features = datasets.Features({
         "image": datasets.Image(),
         "description": datasets.Value("string"),
         "label": datasets.features.ClassLabel(names=_NAMES),
         "file_name": datasets.Value("string"),
      })
      keys = ("image", "label")
      
      return datasets.DatasetInfo(
         description=_DESCRIPTION,
         features=features,
         supervised_keys=keys,
         homepage=_HOMEPAGE,
         citation=_CITATION,
      )
      
      
   def _split_generators(self, dl_manager):      
      train_files = []
      train_idx = []
       
      test_files = []
      test_idx = []
      
      # Download images
      img_data_files = os.path.join(dl_manager.download_and_extract(_URLS["image_urls"]), "images")  # skip _MACOSX dir
      text_data_files = os.path.join(dl_manager.download_and_extract(_URLS["text_urls"]), "text")  # skip _MACOSX dir
      #logger.info(f"text_data_files: {text_data_files}")
      #logger.info(f"text_data_files: {text_data_files[10]}")

       
      img_path_files = sorted(glob.glob(os.path.join(img_data_files, "*", "*.jpg")))
      text_path_files = sorted(glob.glob(os.path.join(text_data_files, "*", "*.txt")))
       
      for img, text in zip(img_path_files, text_path_files):
         img_idx = _IMGNAME2ID[os.path.basename(img)]
         # Sanity check to ensure that pairs of text and image are correct
         if os.path.basename(img).replace(".jpg", "") == os.path.basename(text).replace(".txt", ""):
            if img_idx in _TRAIN_IDX_SET:
               train_files.append((img, text))
               train_idx.append(img_idx)
            else:
               test_files.append((img, text))
               test_idx.append(img_idx)
               
      return [
                 datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                       "files": train_files,
                       "image_idx": train_idx
                    }
                 ),
                 datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                       "files": test_files,
                       "image_idx": test_idx
                    }
                 )
      ]
      
      
   def _generate_examples(self, files, image_idx):
   
      for i, path in enumerate(files):
         file_name = os.path.basename(path[0])
         if file_name.endswith(".jpg"):
            yield i, {
               "image": path[0],
               "description": open(path[1], "r").read(),
               "label": _ID2LABEL[_IMGID2CLASSID[image_idx[i]]],
               "file_name": file_name,
            }
       