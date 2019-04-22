import numpy as np
class Evalutions():
    def __init__(self,pred,gt,classes):
        if type(pred)!=np.narray:
            pred=np.array(pred)
        if type(gt)!=np.array:
            gt=np.array(gt)
        self.tp = 0
        self.fn = 0
        self.fp = 0
        self.tn = 0
        self.classes = classes
        for class_ in classes:
            index_ = classes.index(class_)
            tp_ = ((pred == index_)&(gt == index_)).sum()
            self.tp += tp_
            fn_ = ((pred != index_)&(gt == index_)).sum()
            self.fn += fn_
            fp_ = ((pred == index_)&(gt != index_)).sum()
            self.fp += fp_
            tn_ = ((pred != index_)&(gt != index_)).sum()
            self.tn += tn_
            setattr(self,class_,Evalution(tp_,fn_,fp_,tn_))
        setattr(self,'average',Evalution(self.tp,self.fn,self.fp,self.tn))

    def __repr__(self):
        splitline_str = '*'*230
        classesline_str = ' '*15 + ' |Aveg|    ' + ''.join(['|{}|     '.format(self.classes[i]) for  i in range(len(self.classes))])
        preline_str = 'precision: \t'+ '{:0.4f} '.format(getattr(self,'average').precision()) +''.join([' {:0.4f} '.format(getattr(self,self.classes[i]).precision()) for i in range(len(self.classes))])
        recline_str = 'recall: \t'+ '{:0.4f} '.format(getattr(self,'average').recall()) +''.join([' {:0.4f} '.format(getattr(self,self.classes[i]).recall()) for i in range(len(self.classes))])
        acurline_str = 'accuracy: \t'+ '{:0.4f} '.format(getattr(self,'average').accuracy()) +''.join([' {:0.4f} '.format(getattr(self,self.classes[i]).accuracy()) for i in range(len(self.classes))])
        f1score_str = 'f1_score: \t'+ '{:0.4f} '.format(getattr(self,'average').f1_score()) +''.join([' {:0.4f} '.format(getattr(self,self.classes[i]).f1_score()) for i in range(len(self.classes))])
        
        return splitline_str+'\n'+classesline_str+'\n'+preline_str+'\n'+recline_str+'\n'+acurline_str+'\n'+f1score_str+'\n'+splitline_str
    
    def __dir__(self):
        dir_ = ['average']
        dir_.extend(self.classes)
        return dir_

    def writelog(self,writer,key='average',path='',global_step = None):
        #write.add_scalar('train/accuracy',accuracy_,global_step=iterations_)
        if key in dir(self):
            writer.add_scalar(path+'/{}/precision'.format(key),getattr(self,key).precision(),global_step=global_step)
            writer.add_scalar(path+'/{}/recall'.format(key),getattr(self,key).recall(),global_step=global_step)
            writer.add_scalar(path+'/{}/accuracy'.format(key),getattr(self,key).accuracy(),global_step=global_step)
            writer.add_scalar(path+'/{}/f1_score'.format(key),getattr(self,key).f1_score(),global_step=global_step)
        elif key == 'ALL':
            for attr_ in dir(self):
                writer.add_scalar(path+'/{}/precision'.format(attr_),getattr(self,attr_).precision(),global_step=global_step)
                writer.add_scalar(path+'/{}/recall'.format(attr_),getattr(self,attr_).recall(),global_step=global_step)
                writer.add_scalar(path+'/{}/accuracy'.format(attr_),getattr(self,attr_).accuracy(),global_step=global_step)
                writer.add_scalar(path+'/{}/f1_score'.format(attr_),getattr(self,attr_).f1_score(),global_step=global_step)
        else:
            writer.add_scalar(path+'/{}/precision'.format('average'),getattr(self,'average').precision(),global_step=global_step)
            writer.add_scalar(path+'/{}/recall'.format('average'),getattr(self,'average').recall(),global_step=global_step)
            writer.add_scalar(path+'/{}/accuracy'.format('average'),getattr(self,'average').accuracy(),global_step=global_step)
            writer.add_scalar(path+'/{}/f1_score'.format('average'),getattr(self,'average').f1_score(),global_step=global_step)






class Evalution():
    def __init__(self,tp,fn,fp,tn):
        self.tp = tp
        self.fn = fn
        self.fp = fp
        self.tn = tn
    
    def precision(self):
        return self.tp/(self.tp + self.fp)

    def recall(self):
        return self.tp/(self.tp + self.fn)

    def accuracy(self):
        return (self.tp + self.tn)/(self.tn+self.tp+self.fn+self.fp)

    def f1_score(self):
        return 2*self.tp/(2*self.tp+self.fn+self.fp)