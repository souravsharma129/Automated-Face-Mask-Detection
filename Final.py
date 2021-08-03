from tkinter import *
from PIL import Image,ImageTk
import cv2
import os
import numpy as np
import mysql.connector as mc
from tkinter import filedialog as fd

class FaceDetection:
        fc_path = ''
        mc_path = ''
        img_path = ''
        def getpath(self,a,b,c):
            FaceDetection.fc_path=a+"/"+"haarcascade_frontalface_default.xml"
            FaceDetection.mc_path=b+"/"+"Mouth.xml"
            FaceDetection.img_path = c
        def printvalue(self):
            print(self.fc_path)
            print(self.mc_path)
            print(self.img_path)                
        def faceDetect(self,sap_entry):
            sapid=sap_entry
            print(FaceDetection.fc_path)
            faceCascade = cv2.CascadeClassifier(FaceDetection.fc_path)
            count=0
            pathDir=os.path.join(FaceDetection.img_path+'/','FaceData/')
            os.mkdir(pathDir)
            pathfinalDir = os.path.join(pathDir,sapid)
            os.mkdir(pathfinalDir)
            cap = cv2.VideoCapture(0)
            cap.set(3,640)
            cap.set(4,480)
            while True: 
                
                success,img = cap.read()
                imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces=faceCascade.detectMultiScale(imgGray,1.3,6)
            
                for (x,y,w,h) in faces:
                    cv2.putText(imgGray,str(count),(120,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
                    cv2.rectangle(imgGray,(x,y),(x+w,y+h),(0,0,255),2)
                
                
                path=pathDir+"/"+sapid+'/'+sapid+"."+str(count)+".jpg"
                cv2.imwrite(path,imgGray)
                count=count+1
                cv2.imshow("Frame",imgGray) 
                if (cv2.waitKey(15) & 0xFF == ord('q')) or count==50: 
                    break
                
            cap.release()
            cv2.destroyAllWindows()
        def maskDetect(self,sap_entry):
            sapid = sap_entry
            faceCascade = cv2.CascadeClassifier(FaceDetection.fc_path)
            mouthCascade = cv2.CascadeClassifier(FaceDetection.mc_path)
            count=0
            pathDir=os.path.join(FaceDetection.img_path,'MaskData')
            os.mkdir(pathDir)
            cap = cv2.VideoCapture(0)
            cap.set(3,640)
            cap.set(4,480)
            while True: 
                
                success,img = cap.read()
                imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces=faceCascade.detectMultiScale(img,1.3,5)
                for (x,y,w,h) in faces: 
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                    gray=imgGray[y:y+h,x:x+w]
                    roi=img[y:y+h,x:x+w]
                    mouth=mouthCascade.detectMultiScale(gray,2.245,6)
                    print(mouth)
                    if(len(mouth)==0):
                        if(len(faces)!=0):
                            cv2.putText(img,"Mask Found",(120,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
                    else:
                        cv2.putText(img,"Mask Not Found",(120,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
                    for (x1,y1,w1,h1) in mouth:
                        cv2.rectangle(roi,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)
                        break
                    path=pathDir+'/'+sapid+"."+str(count)+".jpg"
                    cv2.imwrite(path,gray)
                    count=count+1
                cv2.imshow("Frame",img) 
                if (cv2.waitKey(1) & 0xFF == ord('q')) or count==50: 
                    break
                
            cap.release()
            cv2.destroyAllWindows()

class FaceTraining(FaceDetection):
    model=cv2.face.LBPHFaceRecognizer_create()
    def trainingData(self): 
        
        faceCascade = cv2.CascadeClassifier(FaceDetection.fc_path)
        path=os.path.join(FaceDetection.img_path+'/','FaceData/')
        trainingData=[]
        labels=[]
        for f in os.listdir(path):
            label=int(f)
            dirPath=path+f
            for f1 in os.listdir(dirPath):
                imgPath=dirPath+"/"+f1
                images=cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)
                trainingData.append(images)
                labels.append(label)
        print(len(labels))
        print(len(trainingData))
        
        
        self.model.train(trainingData,np.array(labels))
        
        print("Successfully trained")


class MaskTraining(FaceDetection):
    # mouthCascade = cv2.CascadeClassifier(mc_path)
    def bubbleSort(ls,label):
        for i in range (0,len(ls)):
            for j in range (0,len(ls)):
                if(ls[i]>ls[j]):
                    temp=ls[i]
                    ls[i]=ls[j]
                    ls[j]=temp
                    
                    temp1=label[i]
                    label[i]=label[j]
                    label[j]=temp1
        return ls[0],label[0]
    
    def compareImage(cmp):
        verifyPath=os.path.join(FaceDetection.img_path,'MaskData/')
        ls=[]
        label=[]
        for f in os.listdir(verifyPath):
            imgPath=verifyPath+f
            img=cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE) 
            histr = cv2.calcHist([img],[0],None,[256],[0,256]) 
            histr1 = cv2.calcHist([cmp],[0],None,[256],[0,256]) 
            cv2.normalize(histr,histr,0,255,cv2.NORM_MINMAX)
            cv2.normalize(histr1,histr1,0,255,cv2.NORM_MINMAX)
            x=cv2.compareHist(histr,histr1,cv2.HISTCMP_CORREL)
            ls.append(x)
            f=f.split(".")
            f=int(f[0])
            label.append(f)
            result,lb=MaskTraining.bubbleSort(ls,label)
        return result,lb

class FaceRecognition(FaceTraining,MaskTraining):
    
    def fetchData(x,login1,pass1):
        print(x)
        var = mc.connect(user = login1, password = pass1,host='localhost')
        mycursor=var.cursor()
        query1= ("show databases")
        mycursor.execute(query1)
        db = mycursor.fetchall()
        for i in db:
            if i[0]=="information_schema" or i[0]=="mysql" or i[0]=="performance_schema" and i[0]=="admin":
                continue
            db1=i
        query2= ("use %s"%db1)
        mycursor.execute(query2)
        query3= ("show tables")
        mycursor.execute(query3)
        si = mycursor.fetchall()
        print(si)
    
        query= ("SELECT * FROM %s WHERE sapid = %s"%(si[0][0],x))
    
        mycursor.execute(query)
        result=mycursor.fetchall()
        for f in result:
            return f
    def recognition(self,login1,pass1):
        faceCascade = cv2.CascadeClassifier(FaceDetection.fc_path)
        mouthCascade = cv2.CascadeClassifier(FaceDetection.mc_path)
        FaceTraining.trainingData(self)
        cap = cv2.VideoCapture(0)
        while True: 
            success,img = cap.read()
            imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=faceCascade.detectMultiScale(imgGray,1.3,5)
            mouth=mouthCascade.detectMultiScale(imgGray,1.3,5)
            
            for(x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                gray=imgGray[y:y+h,x:x+w]
                roi=img[y:y+h,x:x+w]
                if(len(mouth)==0):
                    if(len(faces)!=0):
                        # cv2.putText(img,"Mask Found",(120,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
                        x1,label=MaskTraining.compareImage(roi)
                        x1=x1*100
                        print("x1=",x1)
                        data=FaceRecognition.fetchData(label,login1,pass1)
                        print(data)
                        if(x1>75):
                            cv2.putText(img,"Match Found",(120,180),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
                            for i in range(0,len(data)):
                                cv2.putText(img,str(data[i]),(120,200+(i*20)),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
#                            cv2.putText(img,str(data[1]),(120,225),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
#                            cv2.putText(img,str(data[2]),(120,245),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
#                            cv2.putText(img,str(data[3]),(120,265),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
                        else:
                            cv2.putText(img,"No Match Found",(120,180),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
                else:
                    # cv2.putText(img,"Mask Not Found",(120,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
                    # FaceDetection.faceDetect()
                    x1, result=self.model.predict(imgGray[y:y + h, x:x + w])
                    #print(result)
                    print(result)
                    print(x1)
                    data=FaceRecognition.fetchData(x1,login1,pass1)
                    
                    if(result <= 70):
                        cv2.putText(img,"Match Found",(120,180),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
                        for i in range(0,len(data)):
                            cv2.putText(img,str(data[i]),(120,200+(i*20)),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
#                           
#                        cv2.putText(img,str(data[0]),(120,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
#                        cv2.putText(img,str(data[1]),(120,225),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
#                        cv2.putText(img,str(data[2]),(120,245),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
#                        cv2.putText(img,str(data[3]),(120,265),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
#            
                    elif(result> 70):
                        cv2.putText(img,"No Match Found",(120,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
                    else:
                        cv2.putText(img,"Face Not Detected",(100,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
        
            cv2.imshow('W',img)
            if(cv2.waitKey(1)==13):
                    break
        
        cap.release()
        cv2.destroyAllWindows()

obj1 = FaceRecognition()


root = Tk()

root.geometry("760x570")
root.resizable(False,False)
root.title("Face Recogniser")

image = Image.open("photo.jpg")
photo = ImageTk.PhotoImage(image)

photo_label = Label(image=photo)
photo_label.place(x=0,y=0,relwidth=1,relheight=1)

# frame = LabelFrame(root, text='buttons', padx=25, pady=25)
# frame.grid(row=0,column=0,padx=10, pady=50)


def putData(l1,array,frame,login1,pass1):
    l2=[]
    for i in range(0,len(l1)):
        l2.append(str(l1[i].get()))
    #array = attributes , l2 = data
    var = mc.connect(user = login1, password = pass1,host='localhost')
    mycursor=var.cursor()
    query1= ("show databases")
    mycursor.execute(query1)
    db = mycursor.fetchall()
    for i in db:
        if i[0]=="information_schema" or i[0]=="mysql" or i[0]=="performance_schema" and i[0]=="admin":
            continue
        db1=i
    query2= ("use %s"%db1)
    mycursor.execute(query2)
    query3= ("show tables")
    mycursor.execute(query3)
    si = mycursor.fetchall()
    
    query5 = ("insert into %s (%s) values ('%s')"%(si[0][0],array[0],l2[0]))
    mycursor.execute(query5)
    var.commit()
    for i in range(1,len(array)):
        query6 = ('update %s set %s="%s" where %s="%s"'%(si[0][0],array[i],l2[i],array[0],l2[0]))
        mycursor.execute(query6)
        var.commit()
    frame.destroy()
    frame = LabelFrame(root, text='add new user', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=10, pady=50)

    
    for i in range(0,len(array)):
        label8=Label(frame,text=array[i])
        label8.grid(row=i,column=0)
        
        tBox9=Entry(frame,width=50,borderwidth=5)
        tBox9.grid(row=i,column=1)
        
        l1.append(tBox9)
    
    added = Label(frame,text='User added Successfully',fg='green')
    button8=Button(frame,text="Add Another User",command=lambda:add_data(frame,login1,pass1))
    button9=Button(frame,text="Quit",command=lambda:back1(frame,login1,pass1))
    added.grid(row=len(array)+1,column=1)
    button8.grid(row=len(array)+2,column=1)
    button9.grid(row=len(array)+3,column=1)
    
    

def add_data(frame,login1,pass1):
    frame.destroy()
    frame = LabelFrame(root, text='add new user', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=10, pady=50)
    
    var = mc.connect(user = login1, password = pass1,host='localhost')

    mycursor=var.cursor()
        
    query1= ("show databases")
    mycursor.execute(query1)
    db = mycursor.fetchall()
    for i in db:
        if i[0]=="information_schema" or i[0]=="mysql" or i[0]=="performance_schema" and i[0]=="admin":
            continue
        db1=i
    query2= ("use %s"%db1)
    mycursor.execute(query2)
    query3= ("show tables")
    mycursor.execute(query3)
    si = mycursor.fetchall()
    query4= ("describe %s"%si[0][0])
    # val = (db[1][0],si[0][0])
    mycursor.execute(query4)
    dt = mycursor.fetchall()
    array = []
    l1=[]
    for i in dt:
        array.append(i[0])
    
    for i in range(0,len(array)):
        label8=Label(frame,text=array[i])
        label8.grid(row=i,column=0)
        
        tBox9=Entry(frame,width=50,borderwidth=5)
        tBox9.grid(row=i,column=1)
        
        l1.append(tBox9)
        
    button8=Button(frame,text="SUBMIT",command=lambda:putData(l1,array,frame,login1,pass1))
    button9=Button(frame,text="Quit",command=lambda:back1(frame,login1,pass1))
    button8.grid(row=len(array),column=1)
    button9.grid(row=len(array)+1,column=1)

def train_with_mask(frame,sap_entry,login1,pass1):
    sap1=sap_entry
    frame.destroy()
    frame = LabelFrame(root, text='training with mask', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=250, pady=50)
    button1 = Button(frame,text="open camera",width=20,height=10,borderwidth=5,command=lambda:obj1.maskDetect(sap1))
    button1.grid(row=0,column=0,padx=20,pady=0)
    button1 = Button(frame,text="Main menu",width=20,height=2,borderwidth=5,command=lambda:back1(frame,login1,pass1))
    button1.grid(row=1,column=0,padx=20,pady=2)


def train_without_mask(frame,sap_entry,login1,pass1):
    sap1=sap_entry.get()
    frame.destroy()
    frame = LabelFrame(root, text='training without mask', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=250, pady=50)
    button1 = Button(frame,text="open camera",width=20,height=10,borderwidth=5,command=lambda:obj1.faceDetect(sap1))
    button1.grid(row=0,column=0,padx=20,pady=0)
    button1 = Button(frame,text="Done",width=20,height=2,borderwidth=5,command=lambda:train_with_mask(frame,sap1,login1,pass1))
    button1.grid(row=1,column=0,padx=20,pady=2)

def sapid_input(frame,login1,pass1):
    frame.destroy()
    frame = LabelFrame(root, text="Input your sapid",padx=25, pady=25)
    note = Label(frame, text="Note : Enter your sapid mentioned in database !!")
    sap_entry = Entry(frame,width=50, borderwidth=5)
    enter_button = Button(frame, text="Enter",command=lambda:train_without_mask(frame, sap_entry,login1,pass1))
    frame.grid(row=0,column=0,padx=200,pady=150)
    note.grid(row=1,column=0,padx=20,pady=0)
    sap_entry.grid(row=2,column=0,padx=20,pady=50)
    enter_button.grid(row=3,column=0,padx=20,pady=0)
    
def back1(frame,login1,pass1):
    frame.destroy()
    frame = LabelFrame(root, text='Main Menu',fg='green', padx=50, pady=25)
    frame.grid(row=0,column=0,padx=10, pady=50)
    button1 = Button(frame,text="Add user to Database",width=20,height=10,borderwidth=5,command=lambda:add_data(frame,login1,pass1))
    button1.grid(row=0,column=0,padx=20,pady=20)

    
    button2 = Button(frame,text="Train Images",width=20,height=10,borderwidth=5,command=lambda:sapid_input(frame,login1,pass1))
    button2.grid(row=0,column=1,padx=20,pady=20)

    button3 = Button(frame,text="Recogniser",width=20,height=10,borderwidth=5,command=lambda:obj1.recognition(login1,pass1))
    button3.grid(row=0,column=2,padx=20,pady=20)
    
def back(frame,name,uname,login1,pass1):
    var = mc.connect(user = login1, password = pass1,host='localhost',database='admin')

    mycursor=var.cursor()
    query= ("SELECT path_dir,path_ff,path_m FROM credentials WHERE username = '%s'"%uname)
    

    mycursor.execute(query)
    result=mycursor.fetchone()
    pd = result[0]
    fc_path= result[1]
    mc_path= result[2]
    obj = FaceDetection()
    obj.getpath(fc_path,mc_path,pd)
    obj.printvalue()
    
    frame.destroy()
    
    
    frame = LabelFrame(root, text='Welcome  '+str(name[0]),fg='green', padx=50, pady=25)
    frame.grid(row=0,column=0,padx=10, pady=50)
    button1 = Button(frame,text="Add user to Database",width=20,height=10,borderwidth=5,command=lambda:add_data(frame,login1,pass1))
    button1.grid(row=1,column=0,padx=20,pady=20)

    
    button2 = Button(frame,text="Train Images",width=20,height=10,borderwidth=5,command=lambda:sapid_input(frame,login1,pass1))
    button2.grid(row=1,column=1,padx=20,pady=20)

    button3 = Button(frame,text="Recogniser",width=20,height=10,borderwidth=5,command=lambda:obj1.recognition(login1,pass1))
    button3.grid(row=1,column=2,padx=20,pady=20)
    
    button4 = Button(frame, text="Logout", width=5,height=1,borderwidth=3,command=lambda:login(frame))
    button4.grid(row=0,column=3)
  

def verify(tBox4,tBox5,frame,dbu,dbp):
    login1=dbu.get()
    pass1=dbp.get()
    uname=tBox4.get()
    pwd=tBox5.get()
    var = mc.connect(user = login1, password = pass1,host='localhost',database='admin')

    mycursor=var.cursor()
    
    query= ("SELECT name FROM credentials WHERE username = %s and password= %s")
    val=(uname,pwd)

    mycursor.execute(query,val)
    result=mycursor.fetchone()
    print(login1)
    if result!=None :
        back(frame,result,uname,login1,pass1)
    else :
        frame.destroy()
        frame = LabelFrame(root, text='Login', padx=25, pady=25)
        frame.grid(row=0,column=0,padx=300, pady=100)
        loginid = Label(frame, text="Username")
        loginid_entry = Entry(frame)
        password = Label(frame,text="Password") 
        
        password_entry = Entry(frame,show="*")
        check_var = IntVar()
        check_show_psw = Checkbutton(frame, text = "Show Password", variable = check_var,onvalue = 1, offvalue = 0, height=2,width = 15, command = lambda:show_hide_psd(password_entry,check_var))
        databaseid = Label(frame,text="Database Username")
        databasepassword = Label(frame,text="Database Password")
        databaseid_entry = Entry(frame)
        databasepassword_entry = Entry(frame,show="*")
        check_var1 = IntVar()
        check_show_psw1 = Checkbutton(frame, text = "Show Password", variable = check_var1,onvalue = 1, offvalue = 0, height=2,width = 15, command = lambda:show_hide_psd(databasepassword_entry,check_var1))
        
        notgood = Label(frame,text = "Wrong Username or Password",fg='red')
        submit= Button(frame, text="Login",command=lambda:verify(loginid_entry,password_entry,frame,databaseid_entry,databasepassword_entry))
        noacc = Label(frame,text="Didn't have an account")
        signupbutton = Button(frame,text="Click here",command=lambda:SignUp(frame))
        
        loginid.grid(row=0,column=0)
        loginid_entry.grid(row=1,column=0)
        password.grid(row=2,column=0)
        password_entry.grid(row=3,column=0)
        databaseid.grid(row=5,column=0)
        databaseid_entry.grid(row=6,column=0)
        databasepassword.grid(row=7,column=0)
        databasepassword_entry.grid(row=8,column=0)
        check_show_psw.grid(row=4,column=0)
        check_show_psw1.grid(row=9,column=0)
        notgood.grid(row=10,column=0)
        submit.grid(row=11,column=0,pady=15)
        noacc.grid(row=12,column=0)
        signupbutton.grid(row=13,column=0)
    

def show_hide_psd(entry_psw,check_var):
    if(check_var.get()):
        entry_psw.config(show="")
    else:
        entry_psw.config(show="*")

def nextbuttonc1(frame,fullname_entry,loginid_entry,password_entry,login1,pass1):
    frame.destroy()
    
    frame = LabelFrame(root, text='Configration', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=300, pady=150)
    
     
    tablebutton = Label(frame,text="Table Created",fg='green')
    
    pathcreate = Label(frame,text="Select path to save images.")
    pathbutton = Button(frame,text="Add path!!",command=lambda:path(frame,fullname_entry,loginid_entry,password_entry,login1,pass1))
    
    pathcreate.grid(row=1,column=0)
    pathbutton.grid(row=1,column=1)
    
    
    tablebutton.grid(row=0,column=0,pady=10)

def collect(dbname,login1,pass1,tbname,al,atl,frame,fullname_entry,loginid_entry,password_entry):
    l1=[]
    l2=[]
    for i in range(0,len(al)):
        l1.append(str(al[i].get()))
        l2.append(str(atl[i].get()))
    var = mc.connect(user=login1,password=pass1,host='localhost',database=dbname)
    var1 = var.cursor()
    var1.execute("create table %s (d int)"%tbname)
    
    for i in range (0,len(al)):
        
        
        var1.execute("alter table %s add %s %s"%(tbname,l1[i],l2[i]))
        
    
    var1.execute("alter table %s drop d"%tbname)
    
    frame.destroy()
    frame = LabelFrame(root, text='Configration', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=50, pady=150)
    
    # primaryks=Label(frame,text="Primary key Selection")
    # for i in range(0,len(al)):
    #     varia = IntVar()
    #     c1 = Checkbutton(frame, text=l1[i], variable=varia, onvalue=1, offvalue=0)
    #     c1.grid(row=i,column=0)
    
    frame.destroy()
    frame = LabelFrame(root, text='Configration', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=50, pady=150)
    
    success = Label(frame,text="Kudos!! Successfulley configurd database and table",fg='green')
    success.grid()
    
    sb = Button(frame,text="Next",command=lambda:nextbuttonc1(frame,fullname_entry,loginid_entry,password_entry,login1,pass1))
    sb.grid(row=1,column=0)
    

    
    
def atncreate(dbname,login1,pass1,tbname,tbe,frame,fullname_entry,loginid_entry,password_entry):
    atnname=tbe.get()
    
    atnname=int(atnname)
    
    frame.destroy()
    
    frame = LabelFrame(root, text='Configration', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=50, pady=150)
    
    db = Label(frame,text="Database Name:")
    tb = Label(frame,text="Table Name:")
    atn = Label(frame,text="Number of attributes in Table:")
    
    db.grid(row=0,column=0)
    tb.grid(row=1,column=0)
    atn.grid(row=2,column=0)
    
    
    dbe = Entry(frame)
    tbe = Entry(frame)
    atne = Entry(frame)
    
    dbe.grid(row=0,column=1)
    tbe.grid(row=1,column=1)
    atne.grid(row=2,column=1)
    
    
    dbb = Label(frame,text="Created!!",fg="green")
    
    
    dbb.grid(row=0,column=2)
    
    
    atnentrylist = []
    atndtentrylist = []
    
    for i in range(0,atnname):
        atnlabel = Label(frame,text="Row"+ str(i+1))
        atnentry  = Entry(frame)
        atndt = Label(frame,text="Datatype")
        atndtentry = Entry(frame)
        
    
        atnentrylist.append(atnentry)
        atndtentrylist.append(atndtentry)
        
        atnlabel.grid(row=3+i,column=0)
        atnentry.grid(row=3+i,column=1)
        atndt.grid(row=3+i,column=2)
        atndtentry.grid(row=3+i,column=3)
        
        
    
    submitbutton = Button(frame,text="Submit",command=lambda:collect(dbname,login1,pass1,tbname,atnentrylist,atndtentrylist,frame,fullname_entry,loginid_entry,password_entry))
    submitbutton.grid(row=atnname+3,column=1)
    
    

def tbcreate(dbname,login1,pass1,tbe,frame,fullname_entry,loginid_entry,password_entry):
    tbname=tbe.get()
    
    frame.destroy()
    
    frame = LabelFrame(root, text='Configration', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=300, pady=150)
    
    db = Label(frame,text="Database Name:")
    tb = Label(frame,text="Table Name:")
    atn = Label(frame,text="Number of attributes in Table:")
    
    db.grid(row=0,column=0)
    tb.grid(row=1,column=0)
    atn.grid(row=2,column=0)
    
    
    dbe = Entry(frame)
    tbe = Entry(frame)
    atne = Entry(frame)
    
    dbe.grid(row=0,column=1)
    tbe.grid(row=1,column=1)
    atne.grid(row=2,column=1)
    
    
    dbb = Label(frame,text="Created!!",fg="green")
    
    atnb = Button(frame,text="Submit",command=lambda:atncreate(dbname,login1,pass1,tbname,atne,frame,fullname_entry,loginid_entry,password_entry))
    
    dbb.grid(row=0,column=2)
    
    atnb.grid(row=2,column=2)

def dbcreate(login1,pass1,dbe,frame,fullname_entry,loginid_entry,password_entry):
    dbname=dbe.get()
    var = mc.connect(user=login1,password=pass1,host='localhost')
    var1 = var.cursor()
    
    var1.execute("create database %s"%dbname)
    frame.destroy()
    
    frame = LabelFrame(root, text='Configration', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=300, pady=150)
    
    db = Label(frame,text="Database Name:")
    tb = Label(frame,text="Table Name:")
    atn = Label(frame,text="Number of attributes in Table:")
    
    db.grid(row=0,column=0)
    tb.grid(row=1,column=0)
    atn.grid(row=2,column=0)
    
    
    dbe = Entry(frame)
    tbe = Entry(frame)
    atne = Entry(frame)
    
    dbe.grid(row=0,column=1)
    tbe.grid(row=1,column=1)
    atne.grid(row=2,column=1)
    
    
    dbb = Label(frame,text="Created!!",fg="green")
    tbb = Button(frame,text="Create!!",command=lambda:tbcreate(dbname,login1,pass1,tbe,frame,fullname_entry,loginid_entry,password_entry))
    
    
    dbb.grid(row=0,column=2)
    tbb.grid(row=1,column=2)
    
    
    

def attributeselection(le,pe,frame,fullname_entry,loginid_entry,password_entry):
    login1 = loginid_entry.get()
    pass1 = password_entry.get()
    var = mc.connect(user=login1,password=pass1,host='localhost')
    var1 = var.cursor()
    q1 = ('create database admin')
    var1.execute(q1)
    q2 = ('use admin')
    var1.execute(q2)
    q3 = ('create table credentials (username varchar(20) primary key,password varchar(20),name varchar(20),path_dir varchar(100),path_ff varchar(100), path_m varchar(100),dbusername varchar(10),dbpassword varchar(10))')
    var1.execute(q3)
    var.commit()
    frame.destroy()
    
    frame = LabelFrame(root, text='Configration', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=300, pady=150)
    
    db = Label(frame,text="Database Name:")
    tb = Label(frame,text="Table Name:")
    atn = Label(frame,text="Number of attributes in Table:")
    
    db.grid(row=0,column=0)
    tb.grid(row=1,column=0)
    atn.grid(row=2,column=0)
    
    
    dbe = Entry(frame)
    tbe = Entry(frame)
    atne = Entry(frame)
    
    dbe.grid(row=0,column=1)
    tbe.grid(row=1,column=1)
    atne.grid(row=2,column=1)
    
    
    dbb = Button(frame,text="Create!!",command=lambda:dbcreate(login1,pass1,dbe,frame,fullname_entry,le,pe))
    dbb.grid(row=0,column=2)

def verifydb(frame,fullname_entry,loginid_entry,password_entry):
    frame.destroy()
    
    frame = LabelFrame(root, text='Enter unsername and Password of Database', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=300, pady=150)
    
    
    loginid = Label(frame, text="Username")
    loginid_entry1 = Entry(frame)
    password = Label(frame,text="Password") 
    password_entry2 = Entry(frame,show='*')
    check_var = IntVar()
    check_show_psw = Checkbutton(frame, text = "Show Password", variable = check_var,onvalue = 1, offvalue = 0, height=2,width = 15, command = lambda:show_hide_psd(password_entry,check_var))

    submit= Button(frame, text="Connect",command=lambda:attributeselection(loginid_entry,password_entry,frame,fullname_entry,loginid_entry1,password_entry2))
    
    loginid.grid(row=0,column=0,padx=30)
    loginid_entry1.grid(row=1,column=0,padx=30)
    password.grid(row=2,column=0,padx=30)
    password_entry2.grid(row=3,column=0,padx=30)
    check_show_psw.grid(row=4,column=0)
    
    submit.grid(row=5,column=0,pady=15,padx=30)
    
def confifcomp(frame,fn,li,ps,pathentry,pathentry1,pathentry2,login1,pass1):
    
    p_dir = str(pathentry.get())
    
    p_ff = str(pathentry1.get())
    
    p_m = str(pathentry2.get())
    
    
    var = mc.connect(user=login1,password=pass1,host='localhost',database='admin')
    
    var1 = var.cursor()
    
    query= ("insert into credentials (username,password,name,path_dir,path_ff,path_m,dbusername,dbpassword) values (%s,%s,%s,%s,%s,%s,%s,%s)")
    
    val=(li,ps,fn,p_dir,p_ff,p_m,login1,pass1)
    
    var1.execute(query,val)
    
    var.commit()
    
    
    frame.destroy()
    
    frame = LabelFrame(root, text='Configration', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=300, pady=150)
    
    pathcreated = Label(frame,text="CONFIGRATION COMPLETE",fg='green')
    donelabel = Label(frame,text="Click to go to Login")
    donebutton = Button(frame,text="Login",command=lambda:login(frame))
    pathcreated.grid(row=0,column=0)
    donelabel.grid(row=1,column=0)
    donebutton.grid(row=1,column=1)
    
def openpath(frame,cdir,pe,peg):
    pe['state']='normal'
    pe.delete(0,'end')
    name = fd.askdirectory(initialdir = cdir,title = "Select file")
    pe.insert(0,name)
    pe['state']= 'disabled'
    
    
    
def path(frame,fullname_entry,loginid_entry,password_entry,login1,pass1):
    frame.destroy()
    
    frame = LabelFrame(root, text='Configration', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=100, pady=150)
    
    cdir = os.getcwd()
    button = Button(frame,text="Browse",command=lambda:openpath(frame,cdir,pathentry,pathentry.get()))
    button1 = Button(frame,text="Browse",command=lambda:openpath(frame,cdir,pathentry1,pathentry1.get()))
    button2 = Button(frame,text="Browse",command=lambda:openpath(frame,cdir,pathentry2,pathentry2.get()))
    donebutton = Button(frame,text="Done",command=lambda:confifcomp(frame,fullname_entry,loginid_entry,password_entry,pathentry,pathentry1,pathentry2,login1,pass1))
    donebutton.grid(row=3,column=1)

    text = Label(frame,text="Path for images : ")
    text1 = Label(frame,text="Path for frontal_face classifier : ")    
    text2 = Label(frame,text="Path for mouth classifier : ")
    
    
    pathentry = Entry(frame,width=50,state='normal')
    pathentry1 = Entry(frame,width=50,state='normal')
    pathentry2 = Entry(frame,width=50,state='normal')
    
    pathentry.insert(0,cdir)
    pathentry['state']='disabled'
    
    pathentry1.insert(0,cdir)
    pathentry1['state']='disabled'
    
    pathentry2.insert(0,cdir)
    pathentry2['state']='disabled'
    
    button.grid(row=0,column=2)
    button1.grid(row=1,column=2)
    button2.grid(row=2,column=2)
    pathentry.grid(row=0,column=1,padx=20)
    pathentry1.grid(row=1,column=1,padx=20)
    pathentry2.grid(row=2,column=1,padx=20)
    text.grid(row=0,column=0)    
    text1.grid(row=1,column=0)    
    text2.grid(row=2,column=0)    



def nextbuttonc(frame,loginid_entry,password_entry,fullname_entry):
    frame.destroy()
    
    frame = LabelFrame(root, text='Configration', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=300, pady=150)
    
    tablecreate = Label(frame,text="Create Table.")
    tablebutton = Button(frame,text="Click Here!!",command=lambda:verifydb(frame,fullname_entry,loginid_entry,password_entry))
    
    pathcreate = Label(frame,text="Select path to save images.")
    pathbutton = Button(frame,text="Add path!!")
    
    pathcreate.grid(row=1,column=0)
    pathbutton.grid(row=1,column=1)
    
    tablecreate.grid(row=0,column=0,pady=10)
    tablebutton.grid(row=0,column=1,pady=10)

def signupdone(frame,fullname_entry,loginid_entry,password_entry):
    li = str(loginid_entry.get())
    ps = str(password_entry.get())
    fn = str(fullname_entry.get())
    
    
    frame.destroy()
    
    frame = LabelFrame(root, text='Sign UP', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=300, pady=150)
    
    fullname = Label(frame, text="Enter Your full Name")
    fullname_entry = Entry(frame)
    
    loginid = Label(frame, text="Username")
    loginid_entry1 = Entry(frame)
    
    password = Label(frame,text="Password") 
    password_entry1 = Entry(frame)
    
    
    done = Label(frame,text="Sign Up Done Successfully!!",fg="green")
    
    nextbutton = Button(frame,text="Next",command=lambda:nextbuttonc(frame,li,ps,fn))
    
    loginid.grid(row=1,column=0)
    loginid_entry1.grid(row=1,column=1)
    
    fullname.grid(row=0,column=0)
    fullname_entry.grid(row=0,column=1)
    
    
    password.grid(row=2,column=0)
    password_entry1.grid(row=2,column=1)
    

    done.grid(row=3,column=0,pady=10)
    nextbutton.grid(row=4,column=0)
def SignUp(frame):
    frame.destroy()
    
    frame = LabelFrame(root, text='Sign UP', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=300, pady=150)
    
    fullname = Label(frame, text="Enter Your full Name")
    fullname_entry = Entry(frame)
    
    loginid = Label(frame, text="Username")
    loginid_entry = Entry(frame)
    
    password = Label(frame,text="Password") 
    password_entry = Entry(frame,show='*')
    check_var = IntVar()
    check_show_psw = Checkbutton(frame, text = "Show Password", variable = check_var,onvalue = 1, offvalue = 0, height=2,width = 15, command = lambda:show_hide_psd(password_entry,check_var))


    submit= Button(frame, text="Sign Up",command=lambda:signupdone(frame,fullname_entry,loginid_entry,password_entry))
    
    loginid.grid(row=1,column=0)
    loginid_entry.grid(row=1,column=1)
    
    fullname.grid(row=0,column=0)
    fullname_entry.grid(row=0,column=1)
    
    
    password.grid(row=2,column=0)
    password_entry.grid(row=2,column=1)
    
    submit.grid(row=4,column=0,pady=15)
    check_show_psw.grid(row=3,column=0)
    
def login(frame):
    frame.destroy()
    frame = LabelFrame(root, text='Login', padx=25, pady=25)
    frame.grid(row=0,column=0,padx=300, pady=100)
    loginid = Label(frame, text="Username")
    loginid_entry = Entry(frame)
    password = Label(frame,text="Password") 
    password_entry = Entry(frame,show='*')
    check_var = IntVar()
    check_show_psw = Checkbutton(frame, text = "Show Password", variable = check_var,onvalue = 1, offvalue = 0, height=2,width = 15, command = lambda:show_hide_psd(password_entry,check_var))
    
    databaseid = Label(frame,text="Database Username")
    databasepassword = Label(frame,text="Database Password")
    databaseid_entry = Entry(frame)
    databasepassword_entry = Entry(frame,show="*")
    check_var1 = IntVar()
    check_show_psw1 = Checkbutton(frame, text = "Show Password", variable = check_var1,onvalue = 1, offvalue = 0, height=2,width = 15, command = lambda:show_hide_psd(databasepassword_entry,check_var1))
    
    
    submit= Button(frame, text="Login",command=lambda:verify(loginid_entry,password_entry,frame,databaseid_entry,databasepassword_entry))
    noacc = Label(frame,text="Didn't have an account")
    signupbutton = Button(frame,text="Click here",command=lambda:SignUp(frame))
    
    
    loginid.grid(row=0,column=0)
    loginid_entry.grid(row=1,column=0)
    password.grid(row=2,column=0)
    password_entry.grid(row=3,column=0)
    databaseid.grid(row=5,column=0)
    databaseid_entry.grid(row=6,column=0)
    databasepassword.grid(row=7,column=0)
    databasepassword_entry.grid(row=8,column=0)
    check_show_psw.grid(row=4,column=0)
    check_show_psw1.grid(row=9,column=0)
    submit.grid(row=10,column=0,pady=15)
    noacc.grid(row=11,column=0)
    signupbutton.grid(row=12,column=0)

frame = LabelFrame(root, text='Login', padx=25, pady=25)
frame.grid(row=0,column=0,padx=300, pady=100)
loginid = Label(frame, text="Username")
loginid_entry = Entry(frame)
password = Label(frame,text="Password") 
password_entry = Entry(frame,show='*')
check_var = IntVar()
check_show_psw = Checkbutton(frame, text = "Show Password", variable = check_var,onvalue = 1, offvalue = 0, height=2,width = 15, command = lambda:show_hide_psd(password_entry,check_var))

databaseid = Label(frame,text="Database Username")
databasepassword = Label(frame,text="Database Password")
databaseid_entry = Entry(frame)
databasepassword_entry = Entry(frame,show="*")
check_var1 = IntVar()
check_show_psw1 = Checkbutton(frame, text = "Show Password", variable = check_var1,onvalue = 1, offvalue = 0, height=2,width = 15, command = lambda:show_hide_psd(databasepassword_entry,check_var1))

submit= Button(frame, text="Login",command=lambda:verify(loginid_entry,password_entry,frame,databaseid_entry,databasepassword_entry))
noacc = Label(frame,text="Didn't have an account")
signupbutton = Button(frame,text="Click here",command=lambda:SignUp(frame))



loginid.grid(row=0,column=0)
loginid_entry.grid(row=1,column=0)
password.grid(row=2,column=0)
password_entry.grid(row=3,column=0)
databaseid.grid(row=5,column=0)
databaseid_entry.grid(row=6,column=0)
databasepassword.grid(row=7,column=0)
databasepassword_entry.grid(row=8,column=0)
check_show_psw.grid(row=4,column=0)
check_show_psw1.grid(row=9,column=0)
submit.grid(row=10,column=0,pady=15)
noacc.grid(row=11,column=0)
signupbutton.grid(row=12,column=0)

root.mainloop()