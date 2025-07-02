from tkinter import *
import tkinter.ttk as ttk
import csv

root = Tk()
root.title("Crime")
width = 1366
height = 768
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width / 2) - (width / 2)
y = (screen_height / 2) - (height / 2)
root.geometry("%dx%d+%d+%d" % (width, height, x, y))
root.resizable(0, 0)

TableMargin = Frame(root, width=500)
TableMargin.pack(side=TOP)

scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
scrollbary = Scrollbar(TableMargin, orient=VERTICAL)

tree = ttk.Treeview(TableMargin, columns=("duration","protocol_type","service","flag","src_bytes","dst_bytes"), height=400, selectmode="extended",
                    yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)


scrollbary.config(command=tree.yview)
scrollbary.pack(side=RIGHT, fill=Y)
scrollbarx.config(command=tree.xview)
scrollbarx.pack(side=BOTTOM, fill=X)

tree.heading('duration', text="duration", anchor=W)
tree.heading('protocol_type', text="protocol_type", anchor=W)
tree.heading('service', text="service", anchor=W)
tree.heading('flag', text="flag", anchor=W)
tree.heading('src_bytes', text="src_bytes", anchor=W)

tree.column('#0', stretch=NO, minwidth=0, width=0)
tree.column('#1', stretch=NO, minwidth=0, width=120)
tree.column('#2', stretch=NO, minwidth=0, width=120)
tree.column('#3', stretch=NO, minwidth=0, width=120)
tree.column('#4', stretch=NO, minwidth=0, width=120)
tree.pack()
with open('NSLKDD.csv') as f:
  reader = csv.DictReader(f, delimiter=',')
  for row in reader:
    a1 = row['duration']
    a2 = row['protocol_type']
    a3 = row['service']
    a4 = row['flag']
    a5 = row['src_bytes']
    tree.insert("", 0, values=(a1,a2,a3,a4,a5))
root.mainloop()
