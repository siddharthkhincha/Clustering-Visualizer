# Necessary imports
import customtkinter
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import os


# Import algorithms
from algorithms.affinity_propogation import call_affinity
from algorithms.db_scan import call_dbscan
from algorithms.k_means import call_kmeans


# Import file_readers
from file_readers.csv_to_ndarray import read_csv
from file_readers.xlsx_to_ndarray import read_xlsx
from file_readers.xls_to_ndarray import read_xls
from file_readers.mat_to_ndarray import read_mat

# Default theme
theme = "dark"

# Global variables
FOLDER_PATH = "./Outputs/"


##############################################################################################################
############################ MAIN GUI ########################################################################

def display_ui():

    ############################ HELPER FUNCTIONS ################################################################

    def browseFiles():
        global filename
        filename = filedialog.askopenfilename(
            # initialdir=os.path.expanduser("~"),  # Start at user home directory
            initialdir=os.getcwd(),  # Start at current directory
            title="Select a File",
            filetypes=(
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("Excel (Current) files", "*.xlsx"),
                ("Excel (Legacy) files", "*.xls"),
                ("Matlab files", "*.mat")
            ))
        print(filename)
        file = open(filename, 'r')
        # content = file.read()
        print(algo_clicked.get())
        browse_button.configure(text=filename.split("/")[-1])
        ret_arr.append(filename)
        return filename

    def submit():
        if (filename == ""):
            print("file not uploaded")
        print(filename)

        temp_arr = filename.split(".")

        if temp_arr[1] not in file_types:
            labelNum4 = customtkinter.CTkLabel(
                root, text="Upload appropriate file")
            labelNum3.place(relx=0.3, rely=0.68)
            labelNum4.place(relx=0.4, rely=0.58)
            submit_button.place(relx=0.35, rely=0.8)
            return
        ret_arr.append(filename)
        ret_arr.append(temp_arr[1])

        # Extract file extension
        extension = temp_arr[1]

        # call appropriate file reader
        global input_data
        print(extension)

        if extension == "csv":
            print(type(has_header))
            print(has_header)
            print(type(plot_dimension_choice))
            print(plot_dimension_choice)
            input_data = read_csv(filename, has_header,
                                  True, plot_dimension_choice)
        elif extension == "xlsx":
            input_data = read_xlsx(filename, has_header,
                                   True, plot_dimension_choice)
        elif extension == "xls":
            input_data = read_xls(filename, has_header,
                                  True, plot_dimension_choice)
        elif extension == "mat":
            print("HERE")
            input_data = read_mat(filename, has_header,
                                  True, plot_dimension_choice)
        if (algo_clicked.get() == "affinity clustering"):
            print(algo_clicked.get())
            ret_arr.append(algo_clicked.get())
            root.destroy()  # destroy the window
            affinity_clustering_window()
        elif (algo_clicked.get() == "kmeans"):
            print(algo_clicked.get())
            ret_arr.append(algo_clicked.get())
            root.destroy()
            kmeans_window()
        elif (algo_clicked.get() == "dbscan"):
            ret_arr.append(algo_clicked.get())
            print(algo_clicked.get())
            root.destroy()
            dbscan_window()

    def affinity_clustering_window():

        global top
        top = customtkinter.CTk()

        top.geometry("400x250")
        top.title('Affinity Clustering')

        Parameter1 = customtkinter.CTkLabel(top, text="Damping Factor")
        Parameter1.place(relx=0.2, rely=0.3)

        Parameter2 = customtkinter.CTkLabel(top, text="Max Iterations")
        Parameter2.place(relx=0.2, rely=0.4)

        global entry1, entry2

        entry1 = customtkinter.CTkEntry(master=top,
                                        placeholder_text="> 0 , < 1",
                                        width=120,
                                        height=25,
                                        border_width=2,
                                        corner_radius=10)
        entry1.place(relx=0.5, rely=0.3)
        entry2 = customtkinter.CTkEntry(master=top,
                                        placeholder_text="> 0 , < 1000",
                                        width=120,
                                        height=25,
                                        border_width=2,
                                        corner_radius=10)
        entry2.place(relx=0.5, rely=0.4)

        submit_buttonx = customtkinter.CTkButton(
            top, text="Submit", command=submit2).place(relx=0.35, rely=0.75)

        top.mainloop()

    def kmeans_window():
        global top
        top = customtkinter.CTk()

        top.geometry("400x250")
        top.title('K-Means')

        Parameter1 = customtkinter.CTkLabel(top, text="Number of Clusters")
        Parameter1.place(relx=0.2, rely=0.3)

        Parameter2 = customtkinter.CTkLabel(top, text="Max Iterations")
        Parameter2.place(relx=0.2, rely=0.4)

        global entry1, entry2

        entry1 = customtkinter.CTkEntry(master=top,
                                        placeholder_text=">1, < num_rows",
                                        width=120,
                                        height=25,
                                        border_width=2,
                                        corner_radius=10)
        entry1.place(relx=0.5, rely=0.3)
        entry2 = customtkinter.CTkEntry(master=top,
                                        placeholder_text="> 0 , < 1000",
                                        width=120,
                                        height=25,
                                        border_width=2,
                                        corner_radius=10)
        entry2.place(relx=0.5, rely=0.4)

        submit_buttonx = customtkinter.CTkButton(
            top, text="Submit", command=submit2).place(relx=0.35, rely=0.75)
        top.mainloop()

    def dbscan_window():
        global top
        top = customtkinter.CTk()

        top.geometry("400x250")
        top.title('DB Scan')

        Parameter1 = customtkinter.CTkLabel(top, text="Epsilon")
        Parameter1.place(relx=0.2, rely=0.3)

        Parameter2 = customtkinter.CTkLabel(top, text="Min Points")
        Parameter2.place(relx=0.2, rely=0.4)

        global entry1, entry2

        # e1 = customtkinter.CTkEntry(top).place(relx = 0.5,y = 0.1)
        entry1 = customtkinter.CTkEntry(master=top,
                                        placeholder_text="> 0 , < 1",
                                        width=120,
                                        height=25,
                                        border_width=2,
                                        corner_radius=10)
        entry1.place(relx=0.5, rely=0.3)
        entry2 = customtkinter.CTkEntry(master=top,
                                        placeholder_text="> 0 , < 1000",
                                        width=120,
                                        height=25,
                                        border_width=2,
                                        corner_radius=10)
        entry2.place(relx=0.5, rely=0.4)

        submit_buttonx = customtkinter.CTkButton(
            top, text="Submit", command=submit2).place(relx=0.35, rely=0.75)
        top.mainloop()

    def submit2():

        # call the appropriate algorithms
        if algo_clicked.get() == "affinity clustering":
            call_affinity(input_data, float(entry1.get()), int(entry2.get()))
        elif algo_clicked.get() == "dbscan":
            call_dbscan(input_data, float(entry1.get()), int(entry2.get()))
        elif algo_clicked.get() == "kmeans":
            call_kmeans(input_data, int(entry1.get()), int(entry2.get()))

        top.destroy()
        after_ui(algo_clicked.get())

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


    ############################ BUILDING UI ###################################################################

    file_types = ["csv", "xlsx", "xls", "mat"]

    customtkinter.set_appearance_mode("dark")
    global has_header, plot_dimension_choice
    global ret_arr
    ret_arr = []

    global root
    root = customtkinter.CTk()
    root.geometry('500x300+100+200')
    customtkinter.set_appearance_mode("dark")
    root.title('PLL')

    # algorithm type dropdown

    labelNum1 = customtkinter.CTkLabel(root, text="Choose algorithm")
    labelNum1.place(relx=0.15, rely=0.1)

    algo_clicked = customtkinter.StringVar(value="affinity clustering")

    drop2 = customtkinter.CTkOptionMenu(master=root,
                                        values=["affinity clustering",
                                                "dbscan", "kmeans"],
                                        variable=algo_clicked)
    drop2.place(relx=0.5, rely=0.1)

    DimensionLabel = customtkinter.CTkLabel(root, text="Choose plot dimension")
    DimensionLabel.place(relx=0.15, rely=0.2)

    dimensions = customtkinter.StringVar(value="3-D")

    drop3 = customtkinter.CTkOptionMenu(master=root,
                                        values=["1-D", "2-D", "3-D"],
                                        variable=dimensions)
    drop3.place(relx=0.5, rely=0.2)

    HeaderLabel = customtkinter.CTkLabel(root, text="Data Headers:")
    HeaderLabel.place(relx=0.15, rely=0.3)

    header = customtkinter.StringVar(value="No")

    drop4 = customtkinter.CTkOptionMenu(master=root,
                                        values=["No", "Yes"],
                                        variable=header)
    drop4.place(relx=0.5, rely=0.3)

    if (header.get() == "True"):
        has_header = True
    else:
        has_header = False
    plot_dimension_choice = int(dimensions.get()[:1])

    # browse files button
    labelNum2 = customtkinter.CTkLabel(root, text="Input Data File:")
    labelNum2.place(relx=0.15, rely=0.4)
    browse_button = customtkinter.CTkButton(
        root, text="Browse Files", command=browseFiles)

    browse_button.place(relx=0.5, rely=0.4)
    labelNum3 = customtkinter.CTkLabel(
        root, text="Only .csv, .xlsx, .xls files allowed")
    labelNum3.place(relx=0.3, rely=0.6)
    # submit button

    submit_button = customtkinter.CTkButton(
        root, text="Submit", command=submit)
    submit_button.place(relx=0.25, rely=0.75)
    global num
    num = 0

    submit_button4 = customtkinter.CTkButton(
        root, text="Change Theme", command=change_theme)
    submit_button4.place(relx=0.62, rely=0.85)

    root.mainloop()

##############################################################################################################
############################ OUTPUT VIEWER UI ################################################################


def slideshow(algorithm):

    ############################ HELPER FUNCTIONS ##############################################################

    def forward(img_no, max_image):
        submit_button.configure(text="Reset")
        print(img_no)
        global label
        global button_forward
        global button_back
        label = Label(image=List_images[img_no-1])

        label.grid(row=1, column=0, columnspan=3)

        if (img_no != 1):
            button_back = customtkinter.CTkButton(
                slideshow_root, text="Back", command=lambda: back(img_no-1, max_image))
            button_back.grid(row=5, column=0)

        if (img_no != max_image):
            button_for = customtkinter.CTkButton(
                slideshow_root, text="Next", command=lambda: forward(img_no+1, max_image))
            button_for.grid(row=5, column=2)

        # update slider value
        slider.set(img_no)

    def back(img_no, max_image):

        global label
        global button_forward
        global button_back
        label = Label(image=List_images[img_no - 1])
        label.grid(row=1, column=0, columnspan=3)
        print(img_no)
        if (img_no != 1):
            button_back = customtkinter.CTkButton(
                slideshow_root, text="Back", command=lambda: back(img_no-1, max_image))
            button_back.grid(row=5, column=0)

        if (img_no != max_image):
            button_for = customtkinter.CTkButton(
                slideshow_root, text="Next", command=lambda: forward(img_no+1, max_image))
            button_for.grid(row=5, column=2)

        # update slider value
        slider.set(img_no)


    ############################ BUILDING UI ###################################################################

    slideshow_root = customtkinter.CTk()

    slideshow_root.title("Image Viewer")
    # slideshow_root.geometry("432x450")
    # slideshow should start with full screen
    slideshow_root.attributes('-fullscreen', True)

    if algorithm == "affinity clustering":
        folder_path = FOLDER_PATH + "AffinityPropogations/"
    if algorithm == "dbscan":
        folder_path = FOLDER_PATH + "DBScan/"
    if algorithm == "kmeans":
        folder_path = FOLDER_PATH + "Kmeans/"
    List_images = []
    total_image = len(os.listdir(folder_path))
    print(len(os.listdir(folder_path)))
    for i in range(0, len(os.listdir(folder_path))):
        img_path = folder_path + "output" + str(i) + ".png"
        print(img_path)
        # check if file exists
        if os.path.isfile(img_path):
            img = Image.open(img_path)
            List_images.append(ImageTk.PhotoImage(img))
            img.close()
        else:
            print(f"File {img_path} does not exist.")

    print("All images loaded successfully!")

    # print(List_images)

    submit_button = customtkinter.CTkButton(
        slideshow_root, text="Start", command=lambda: forward(1, total_image))
    submit_button.place(relx=0.35, rely=0.75)

    # add slider
    slider = Scale(slideshow_root, from_=1, to=total_image, orient=HORIZONTAL,
                   command=lambda value: forward(int(value), total_image))
    slider.place(relx=0.3, rely=0.85, relwidth=0.4)

    close_button = customtkinter.CTkButton(
        slideshow_root, text="Close", command=slideshow_root.destroy)

    close_button.place(relx=0.9, rely=0.05)

    slideshow_root.mainloop()


##############################################################################################################
############################ START SLIDESHOW UI ##############################################################

def after_ui(algorithm):

    ############################ HELPER FUNCTIONS ##############################################################

    def show_slideshow(algorithm):
        root2.destroy()
        slideshow(algorithm)

    ############################ BUILDING UI ###################################################################

    global root2
    root2 = customtkinter.CTk()
    root2.geometry('400x200+100+200')

    root2.title('See results')

    slideshow_button = customtkinter.CTkButton(
        root2, text="See slideshow", command=show_slideshow(algorithm)).place(relx=0.35, rely=0.2)

    root2.mainloop()



##############################################################################################################
############################ MAIN FUNCTION ###################################################################

display_ui()
print(ret_arr)
