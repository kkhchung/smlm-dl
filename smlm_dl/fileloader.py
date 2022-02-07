import zipfile
import PIL.Image
import skimage.io
import numpy as np

class ImageFileLoader(object):
    def __init__(self, file_path, slices, stack_to_volume, cache=True):
        super().__init__()
        self.file_path = file_path
        self.slices = slices
        self.stack_to_volume = stack_to_volume
        
    def __len__(self):
        raise NotImplemented()
        
    def set_images_cache(self, cache):
        self.cache = cache
        if self.cache is True:
            images = list()
            for i in range(len(self)):
                images.append(self.getitem__live(i))
            self.images = np.concatenate(images, axis=0)
        
    def getitem__live(self, key):
        raise NotImplemented()
        
    def __getitem__(self, key):
        if self.cache is True:
            return self.images[[key]]
        else:
            return self.getitem__live(key)


class PilImageFileLoader(ImageFileLoader):
    def __init__(self, file_path, slices, stack_to_volume, cache):
        super().__init__(file_path=file_path, slices=slices,
                         stack_to_volume=stack_to_volume, cache=cache)
        
        self.pil_image = PIL.Image.open(self.file_path)
        if self.stack_to_volume is True:
            self.len = 1
        else:
            self.len = getattr(self.pil_image, 'n_frames', 1)
        
        self.set_images_cache(cache)
        if self.cache is True:
            self.pil_image.close()
    
    def __len__(self):
        return self.len
    
    def getitem__live(self, key):
        if self.stack_to_volume is True:
            images = list()
            zs = np.arange(getattr(self.pil_image, 'n_frames', 1))[self.slices[-1]]
            for i in zs:
                self.pil_image.seek(i)
                images.append(np.asarray(self.pil_image)[self.slices[:-1]].astype(np.float32))
            image = np.stack(images, axis=-1)
        else:
            self.pil_image.seek(key)
            image = np.asarray(self.pil_image)[self.slices].astype(np.float32)
        image = image[None,...]
        return image
    
    def __del__(self):
        try:
            self.pil_image.close()
        except:
            pass


class SkImageFileLoader(ImageFileLoader):
    def __init__(self, file_path, slices, stack_to_volume, cache):
        if cache is False:
            print("The file is always loaded to memory with SkImageFileLoader")
        super().__init__(file_path=file_path, slices=slices,
                         stack_to_volume=stack_to_volume, cache=False)
        
        self.images = skimage.io.imread(self.file_path).astype(np.float32)
        
        if self.stack_to_volume is True:
            self.images = np.moveaxis(self.images, 0, -1)
            self.images = self.images[self.slices][None,...]
        else:
            self.images = self.images[(slice(None),)+self.slices][:,...]
            
        self.set_images_cache(False)
    
    def __len__(self):
        return self.images.shape[0]
    
    def getitem__live(self, key):
        return self.images[[key]]


class ZipSkImageFileLoader(ImageFileLoader):
    def __init__(self, file_path, slices, stack_to_volume, cache):
        super().__init__(file_path=file_path, slices=slices,
                         stack_to_volume=stack_to_volume, cache=cache)
        
        self.file_path = file_path
        with zipfile.ZipFile(self.file_path, 'r') as f_zip:
            self.files = [info.filename for info in f_zip.infolist() if not info.is_dir()]
            self.files = sorted(self.files)
            if self.stack_to_volume is True:
                self.len = 1
            else:
                self.len = len(self.files)
            
        self.set_images_cache(cache)
    
    def __len__(self):
        return self.len
    
    def getitem__live(self, key):
        with zipfile.ZipFile(self.file_path, 'r') as f_zip:
            if self.stack_to_volume is True:
                images = list()
                zs = np.arange(len(self.files))[self.slices[-1]]
                for i in zs:
                    filename = self.files[i]
                    with f_zip.open(filename) as f_img:
                        images.append(skimage.io.imread(f_img)[self.slices[:-1]].astype(np.float32))
                image = np.stack(images, axis=-1)
            else:
                filename = self.files[key]
                with f_zip.open(filename) as f_img:
                    image = skimage.io.imread(f_img)[self.slices].astype(np.float32)
        image = image[None,...]
        return image