import tkinter as customtkinter  
from functools import partial  
from tkinter import OptionMenu, StringVar, filedialog
import customtkinter
from tkinter import *
from PIL import ImageTk, Image
import os
import tkinter as customtkinter  
from functools import partial  
from tkinter import OptionMenu, StringVar, filedialog
import customtkinter


theme = "dark"



def display_ui():

    file_types = ["csv", "xlsx", "xls"]
    
    customtkinter.set_appearance_mode("dark")
    
    global ret_arr
    ret_arr = []

    def browseFiles():
        global filename
        filename = filedialog.askopenfilename(initialdir = "/home/techniche",
                                            title = "Select a File",
                                            filetypes = (("Text files",
                                                            "*.*"),
                                                        ("all files",
                                                            "*.*")))
        
        print(filename)
        file = open(filename,'r')
        content=file.read()
        print(algo_clicked.get())
        browse_button.configure(text=filename.split("/")[-1])
        ret_arr.append(filename)
        return filename
        

    def submit():
        if(filename==""):
            print("file not uploaded")
        print(filename)
        
        temp_arr = filename.split(".")
        
        if temp_arr[1] not in file_types:
            labelNum4 = customtkinter.CTkLabel(root, text="Upload appropriate file")
            labelNum3.place(relx = 0.3,rely=0.68)
            labelNum4.place(relx=0.4,rely=0.58)
            submit_button.place(relx=0.35,rely = 0.8)
            return
        ret_arr.append(filename)
        ret_arr.append(temp_arr[1])
        if(algo_clicked.get() == "affinity clustering"):
            print(algo_clicked.get())
            ret_arr.append(algo_clicked.get())
            root.destroy()
            affinity_clustering_window()
        elif(algo_clicked.get()=="kmeans"):
            print(algo_clicked.get())
            ret_arr.append(algo_clicked.get())
            root.destroy()
            kmeans_window()
        elif(algo_clicked.get()=="dbscan"):
            ret_arr.append(algo_clicked.get())
            print(algo_clicked.get())
            root.destroy()
            dbscan_window()
        




    def affinity_clustering_window():

        global top
        top = customtkinter.CTk()  
    
        top.geometry("400x250")  
        top.title('Affinity Clustering')

        Parameter1 = customtkinter.CTkLabel(top, text = "Parameter1")
        Parameter1.place(relx = 0.2,rely = 0.3)  
        
        Parameter2 = customtkinter.CTkLabel(top, text = "Parameter2")
        Parameter2.place(relx = 0.2,rely = 0.4)  
        
        Parameter3 = customtkinter.CTkLabel(top, text = "Parameter3")
        Parameter3.place(relx = 0.2,rely = 0.5)  
        
        global entry1, entry2, entry3
        
        # e1 = customtkinter.CTkEntry(top).place(relx = 0.5,y = 0.1)  
        entry1 = customtkinter.CTkEntry(master=top,
                               placeholder_text="CTkEntry",
                               width=120,
                               height=25,
                               border_width=2,
                               corner_radius=10)
        entry1.place(relx=0.5, rely=0.3)
        entry2 = customtkinter.CTkEntry(master=top,
                               placeholder_text="CTkEntry",
                               width=120,
                               height=25,
                               border_width=2,
                               corner_radius=10)
        entry2.place(relx=0.5, rely=0.4)
        entry3 = customtkinter.CTkEntry(master=top,
                               placeholder_text="CTkEntry",
                               width=120,
                               height=25,
                               border_width=2,
                               corner_radius=10)
        entry3.place(relx=0.5, rely=0.5)

        submit_buttonx = customtkinter.CTkButton(top, text="Submit", command=submit2).place(relx = 0.35, rely = 0.75)
        # ret_arr.append([entry1.get(), entry2.get(), entry3.get()])
        # print(ret_arr)
        top.mainloop() 
    
    def kmeans_window():
        global top
        top = customtkinter.CTk()  
    
        top.geometry("400x250")  
        top.title('K-Means')

        Parameter1 = customtkinter.CTkLabel(top, text = "Parameter1")
        Parameter1.place(relx = 0.2,rely = 0.3)  
        
        Parameter2 = customtkinter.CTkLabel(top, text = "Parameter2")
        Parameter2.place(relx = 0.2,rely = 0.4)  
        
        Parameter3 = customtkinter.CTkLabel(top, text = "Parameter3")
        Parameter3.place(relx = 0.2,rely = 0.5)  
        
        global entry1, entry2, entry3
        
        # e1 = customtkinter.CTkEntry(top).place(relx = 0.5,y = 0.1)  
        entry1 = customtkinter.CTkEntry(master=top,
                               placeholder_text="CTkEntry",
                               width=120,
                               height=25,
                               border_width=2,
                               corner_radius=10)
        entry1.place(relx=0.5, rely=0.3)
        entry2 = customtkinter.CTkEntry(master=top,
                               placeholder_text="CTkEntry",
                               width=120,
                               height=25,
                               border_width=2,
                               corner_radius=10)
        entry2.place(relx=0.5, rely=0.4)
        entry3 = customtkinter.CTkEntry(master=top,
                               placeholder_text="CTkEntry",
                               width=120,
                               height=25,
                               border_width=2,
                               corner_radius=10)
        entry3.place(relx=0.5, rely=0.5)

        submit_buttonx = customtkinter.CTkButton(top, text="Submit", command=submit2).place(relx = 0.35, rely = 0.75)
        # ret_arr.append([entry1.get(), entry2.get(), entry3.get()])
        # print(ret_arr)
        top.mainloop() 

    def dbscan_window():
        global top
        top = customtkinter.CTk()  
    
        top.geometry("400x250")  
        top.title('DB Scan')

        Parameter1 = customtkinter.CTkLabel(top, text = "Parameter1")
        Parameter1.place(relx = 0.2,rely = 0.3)  
        
        Parameter2 = customtkinter.CTkLabel(top, text = "Parameter2")
        Parameter2.place(relx = 0.2,rely = 0.4)  
        
        Parameter3 = customtkinter.CTkLabel(top, text = "Parameter3")
        Parameter3.place(relx = 0.2,rely = 0.5)  
        
        global entry1, entry2, entry3
        
        # e1 = customtkinter.CTkEntry(top).place(relx = 0.5,y = 0.1)  
        entry1 = customtkinter.CTkEntry(master=top,
                               placeholder_text="CTkEntry",
                               width=120,
                               height=25,
                               border_width=2,
                               corner_radius=10)
        entry1.place(relx=0.5, rely=0.3)
        entry2 = customtkinter.CTkEntry(master=top,
                               placeholder_text="CTkEntry",
                               width=120,
                               height=25,
                               border_width=2,
                               corner_radius=10)
        entry2.place(relx=0.5, rely=0.4)
        entry3 = customtkinter.CTkEntry(master=top,
                               placeholder_text="CTkEntry",
                               width=120,
                               height=25,
                               border_width=2,
                               corner_radius=10)
        entry3.place(relx=0.5, rely=0.5)

        submit_buttonx = customtkinter.CTkButton(top, text="Submit", command=submit2).place(relx = 0.35, rely = 0.75)
        # ret_arr.append([entry1.get(), entry2.get(), entry3.get()])
        # print(ret_arr)
        top.mainloop() 

  

    def submit2():
        ret_arr.append(entry1.get());
        ret_arr.append(entry2.get());
        ret_arr.append(entry3.get());
        print(ret_arr)
        top.destroy()
        after_ui()
    
    
    

    global root
    root = customtkinter.CTk()  
    root.geometry('400x200+100+200')  
    customtkinter.set_appearance_mode("dark")
    root.title('PLL')  
    
    
    #algorithm type dropdown
    labelNum1 = customtkinter.CTkLabel(root, text="Choose algorithm")
    labelNum1.place(relx=0.15,rely=0.2)
   
    algo_clicked = customtkinter.StringVar(value="affinity clustering") 
    # algo_clicked.set( "affinity clustering" )
    # drop2 = customtkinter.CTkOptionMenu( root , algo_clicked , *algo_options )

    drop2 = customtkinter.CTkOptionMenu(master=root,
                                       values=[ "affinity clustering", "dbscan", "kmeans"],
                                       variable=algo_clicked)
    drop2.place(relx = 0.5,rely=0.2)

    #browse files button
    labelNum2 = customtkinter.CTkLabel(root, text="Input Data File:")
    labelNum2.place(relx=0.15,rely=0.4)
    browse_button = customtkinter.CTkButton(root,text = "Browse Files",command = browseFiles)
    
    browse_button.place(relx=0.5,rely=0.4)
    labelNum3 = customtkinter.CTkLabel(root, text="Only .csv, .xlsx, .xls files allowed")
    labelNum3.place(relx=0.3,rely=0.6)
    #submit button
    
    submit_button = customtkinter.CTkButton(root, text="Submit", command=submit)
    submit_button.place(relx = 0.25, rely = 0.75)
    global num
    num = 0
   
    def change_theme():
        global num  # add this line
        print(num)
        if num == 0:
            customtkinter.set_appearance_mode("light")
            num = 1
        else:
            customtkinter.set_appearance_mode("dark")
            num = 0
        print(num)

    submit_button4 = customtkinter.CTkButton(root, text="Change Theme", command=change_theme)
    submit_button4.place(relx=0.62, rely=0.85)

    root.mainloop()  
    

def slideshow():
        
    

    def forward(img_no,max_image):
        submit_button.configure(text="Reset")
        print(img_no)
        global label
        global button_forward
        global button_back
        label = Label(image=List_images[img_no-1])
    
        label.grid(row=1, column=0, columnspan=3)
        
        if(img_no!=1):
            button_back = customtkinter.CTkButton(slideshow_root, text="Back", command=lambda: back(img_no-1,max_image))
            button_back.grid(row=5, column=0)

        if(img_no!=max_image):    
            button_for = customtkinter.CTkButton(slideshow_root, text="Next", command=lambda: forward(img_no+1,max_image))
            button_for.grid(row=5, column=2)
    
    
    def back(img_no,max_image):
    
        global label
        global button_forward
        global button_back       
        label = Label(image=List_images[img_no - 1])
        label.grid(row=1, column=0, columnspan=3)
        print(img_no)
        if(img_no!=1):
            button_back = customtkinter.CTkButton(slideshow_root, text="Back", command=lambda: back(img_no-1,max_image))
            button_back.grid(row=5, column=0)

        if(img_no!=max_image):    
            button_for = customtkinter.CTkButton(slideshow_root, text="Next", command=lambda: forward(img_no+1,max_image))
            button_for.grid(row=5, column=2)
        
        
    slideshow_root = customtkinter.CTk()  

    slideshow_root.title("Image Viewer")    
    slideshow_root.geometry("432x450")
    
    folder_path = './DBScan/' 
    List_images = []
    total_image = len(os.listdir(folder_path))
    print(len(os.listdir(folder_path)));
    for i in range(0,len(os.listdir(folder_path))):
        img_path = folder_path + "output"+str(i)+".png"
        img = Image.open(img_path)
        List_images.append(ImageTk.PhotoImage(Image.open(img_path)))
        img.close()

    # print(List_images)
    
    
    submit_button = customtkinter.CTkButton(slideshow_root, text="Start", command=lambda: forward(1,total_image))
    submit_button.place(relx = 0.35, rely = 0.75)
    
    
    slideshow_root.mainloop()

    
def after_ui():
    global root2
    root2 = customtkinter.CTk()  
    root2.geometry('400x200+100+200')  
    
    root2.title('See results')  
   
   
    def show_slideshow():
        root2.destroy()
        slideshow()

    # slideshow_button = customtkinter.CTkButton(root2, text = "See video").place(relx=0.35,rely=0.4)
    
    

    slideshow_button = customtkinter.CTkButton(root2,text = "See slideshow", command = show_slideshow).place(relx = 0.35,rely=0.2)
    
    
    root2.mainloop()  



display_ui()

print(ret_arr)


