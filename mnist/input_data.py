from PIL import Image
import struct
import random
class Imagedata(object):
    class Data(object):
        def __init__(self,image=None,label=None,size=None):
            self.image=image
            self.label=label
            self.size=size
        def toImage(self):
            image=Image.new('L',self.size)
            for x in range(self.size[1]):
                for y in range(self.size[0]):
                    image.putpixel((y,x),self.image[self.size[0]*x+y])
            return image
    def __init__(self,imagefile=None,labelfile=None):
        self.data=[]
    def read_data(self,imagefile=None,labelfile=None):
        if imagefile and labelfile:
            images=[]
            labels=[]
            images,size=read_image_onehot(imagefile,images)
            labels=read_label_onehot(labelfile,labels)
            self.data=[ Imagedata.Data(images[x],labels[x],size) for x in range(len(labels)) ]
    def __getitem__(self,i):
        return self.data[i]
    def next_batch(self,bsize):
        res_image=[]
        res_label=[]
        label_onehot=[0]*10
        for i in range(bsize):
            idx=random.randint(0,len(self.data)-1)
            res_image.append(self.data[idx].image)
            # 对标签进行onehot编码
            label_onehot[self.data[idx].label]=1
            res_label.append(label_onehot)
            label_onehot[self.data[idx].label]=0
        return res_image,res_label
    @property
    def images(self):
        return [self.data[i].image for i in range(len(self.data))]
    @property
    def labels(self):
        label_onehot=[0]*10
        res_labels=[]
        for i in range(len(self.data)):
            label_onehot[self.data[i].label]=1
            res_labels.append(label_onehot)
            label_onehot[self.data[i].label]=0
        return res_labels            

def read_image_onehot(filename,imageList):
    f=open(filename,'rb')
    index=0;
    buf=f.read()
    f.close()

    magic,images,rows,columns=struct.unpack_from(">IIII",buf,index)
    index+=struct.calcsize('>IIII')
    for i in range(images):
        image=[]
        for x in range(rows):
            for y in range(columns):
                image.append(int(struct.unpack_from('>B',buf,index)[0]))
                index+=struct.calcsize('>B')
        imageList.append(image)
    return imageList,(rows,columns)
def read_label_onehot(filename,lableList):
    f=open(filename,'rb')
    index=0
    buf=f.read()
    f.close()

    magic,labels=struct.unpack_from(">ll",buf,index)
    index+=struct.calcsize(">ll")

    labelList=[0]*labels
    for x in range(labels):
        labelList[x]=int(struct.unpack_from(">B",buf,index)[0])
        index+=struct.calcsize('>B')
    return labelList
 
def read_image(filename):
    f=open(filename,'rb')
    index=0;
    buf=f.read()
    f.close()

    magic,images,rows,columns=struct.unpack_from(">IIII",buf,index)
    index+=struct.calcsize('>IIII')
    for i in range(20):
        image=Image.new('L',(columns,rows))

        for x in range(rows):
            for y in range(columns):
                image.putpixel((y,x),int(struct.unpack_from('>B',buf,index)[0]))
                index+=struct.calcsize('>B')
        print("save %d image"%i)
        image.save('test%d.png'%i)
        
def read_label(filename,saveFilename):
    f=open(filename,'rb')
    index=0
    buf=f.read()
    f.close()

    magic,labels=struct.unpack_from(">ll",buf,index)
    index+=struct.calcsize(">ll")

    labelArr=[0]*labels
    for x in range(20):
        labelArr[x]=int(struct.unpack_from(">B",buf,index)[0])
        index+=struct.calcsize('>B')

    save=open(saveFilename,'w')
    save.write(','.join(map(lambda x:str(x),labelArr)))
    save.write('\n')
    save.close()
    print("save labels success")

if __name__=="__main__":
    mnist_train=Imagedata()
    mnist_train.read_data("train-images.idx3-ubyte","train-labels.idx1-ubyte")
    image=mnist_train[0].toImage()
    image.show()
