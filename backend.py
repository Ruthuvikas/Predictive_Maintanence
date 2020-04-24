import os
from flask import *
#import bin_class  
from cnn import cnn_test
from binary_classification import bin_test
from multiclass import multiclass_test
app = Flask(__name__)  

cwd = os.getcwd() 

UPLOAD_FOLDER = os.path.join(cwd,'download')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/',methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/upload')  
def upload():  
    return render_template("mini-upload-form/index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/clients')
def clients():
    return render_template("clients.html")  

@app.route('/partners')
def partners():
    return render_template("partners.html")
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file'] 
        print('file==============',UPLOAD_FOLDER + '/' + f.filename)
        f.save(UPLOAD_FOLDER + '/' + f.filename)
        #f.save(f.filename)  
        return

@app.route('/binary_classification',methods = ['GET','POST'])
def binary_classification():
    ans = bin_test.binary_main()
    return render_template("binaryclass_result.html",ans = ans)    

@app.route('/cnn_predict',methods = ['GET','POST'])
def cnn_predict():
    ans = cnn_test.cnn_main()
    return render_template("cnn_result2.html",ans = ans)

@app.route('/multiclass',methods = ['GET','POST'])
def multiclass():
    ans = multiclass_test.multiclass_main()
    return render_template("multiclass_result.html",ans = ans)

@app.route('/train',methods = ['GET','POST'])
def train():
    return render_template("train.html")


  
if __name__ == '__main__':  
    app.run(debug = True)  