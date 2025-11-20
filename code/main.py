from SIF import run_sif
from SIFdual import run_sifdual
from uSIF import run_usif
from uSIFdual import run_usifdual
from BoW import run_bow
from BoWdual import run_bowdual
from TextCNN import run_textcnn
from Trans import run_trans
from Bilstm import run_bilstm
from TextCNNdual import run_textcnndual
from Transdual import run_transdual
from Bilstmdual import run_bilstmdual
from Lstm import run_lstm
from Lstmdual import run_lstmdual



traindata = "dataset/train.txt"
testdata = "dataset/test.txt"

result_path = "result/"

wordvec =    "model/sg512main.json" #cbow512main.json sg glove
wordvecsub = "model/sg512sub.json" #cbow512sub.json
wordvecmix = "model/sgcombined.json"
wordvecshapevae = "model/glyph_vae_features.json" 

cuda = 5

def cpucode():
    '''
    run_sif(
        word_vec_path=wordvecshapevae,
        data_paths=[traindata, testdata],
        output_csv_path=result_path + "svm/vae-dan-sif.csv", #base不是baseshuang
        repeat=1,
        cuda_device=cuda,
        use_subword=True,
        lr = 1e-4
    )
    
    
    run_usif(
        word_vec_path=wordvecshapevae,
        data_paths=[traindata, testdata],
        output_csv_path=result_path + "svm/vae-dan-usif.csv",
        repeat=1,
        cuda_device=cuda,
        use_subword=True,
        lr = 1e-4
    )
    '''
  
    run_bow(
        train_path=traindata,
        test_path=testdata,
        output_csv_path=result_path + "svm/zi-dan-bow.csv",
        repeat=1,
        cuda_device=cuda,
        use_subword=True, 
        lr = 1e-4
    )
   

    


def basecode(): 
    '''
    run_textcnn(
        train_path=traindata,
        test_path=testdata,
        word_vec_path=wordvecshapevae,
        output_csv_path=result_path + "vae/dan/textCNN-6/textcnn-1e6.csv",
        repeat=5,
        cuda_device=cuda,
        use_subword=True,
        lr = 1e-6,
        savemodel = True,
        savecanshu  = True
    )
    '''
    
    run_trans(
        train_path=traindata,
        test_path=testdata,
        word_vec_path=wordvecmix,
        output_csv=result_path + "buchong/mix-dan-trans-5/transformer-1e5-2.csv",
        repeat=1,
        cuda_device=cuda,
        use_subword=True,
        lr = 1e-5, 
        savemodel = False,
        savecanshu  = False
    )
    '''
    
    
    run_bilstm(
        train_path=traindata,
        test_path=testdata,
        word_vec_path=wordvecshapevae,
        output_csv_path=result_path + "vae/dan/bilstm-6/bilstm-1e6.csv",
        repeat=5,
        cuda_device=cuda,
        use_subword=True,
        lr = 1e-6, 
        savemodel = True,
        savecanshu  = True
    ) 
    '''

    run_lstm(
        train_path=traindata,
        test_path=testdata,
        word_vec_path=wordvecshapevae,
        output_csv_path=result_path + "vae/dan/lstm-6/bilstm-1e6.csv",
        repeat=5,
        cuda_device=cuda,
        use_subword=True,
        lr = 1e-6, 
        savemodel = True,
        savecanshu  = True
    ) 
    

def basecodedual(): 
    '''
    run_textcnndual(
        train_path=traindata,
        test_path=testdata,
        word_vec_path=wordvecshapevae,
        output_csv_path=result_path + "vae/shuang/textCNN-6/textcnn-6.csv",
        repeat=5,
        cuda_device=cuda,
        use_subword=True,
        lr = 1e-6,
        savemodel = True,
        savecanshu  = True
    )
    '''
    
    run_transdual(
        train_path=traindata,
        test_path=testdata,
        word_vec_path=wordvecsub,
        output_csv=result_path + "buchong/zi-dual-trans/transformer-1e6.csv",
        repeat=1,
        cuda_device=cuda,
        use_subword=True,
        lr = 1e-6, 
        savemodel = False,
        savecanshu  = False
    )
    '''
    
    canshucuda = 4
    run_bilstmdual(
        train_path=traindata,
        test_path=testdata,
        word_vec_path=wordvecshapevae,
        output_csv_path=result_path + "canshu/seed2025/seed2025-5.csv",
        repeat=1,
        cuda_device=canshucuda,
        use_subword=True,
        lr = 1e-4,         # 1e-4
        batch_size=64,     # 64
        seed = 2025,   #'random'
        savemodel = True,
        savecanshu  = True
    ) 
    '''
    '''
    run_lstmdual(
        train_path=traindata,
        test_path=testdata,
        word_vec_path=wordvecshapevae,
        output_csv_path=result_path + "vae/shuang/lstm-6/bilstm-1e6.csv",
        repeat=5,
        cuda_device=cuda,
        use_subword=True,
        lr = 1e-6, 
        savemodel = True,
        savecanshu  = True
    ) 
    '''


def cpucodedual():
    '''
    run_sifdual(
        word_vec_path=wordvecshapevae,
        data_paths=[traindata, testdata],
        output_csv_path=result_path + "svm/vae-dual-sif", #base不是baseshuang
        repeat=1,
        cuda_device=cuda,
        use_subword=True,
        lr = 1e-4
    )
    '''
    '''
    run_usifdual(
        word_vec_path=wordvecshapevae,
        data_paths=[traindata, testdata],
        output_csv_path=result_path + "svm/vae-dual-usif.csv",
        repeat=1,
        cuda_device=cuda,
        use_subword=True,
        lr = 1e-4
    )
    '''
    
    run_bowdual(
        train_path=traindata,
        test_path=testdata,
        output_csv_path=result_path + "svm/zi-dual-bow.csv",
        repeat=1,
        cuda_device=cuda,
        use_subword=True, 
        lr = 1e-4
    )
    


if __name__ == "__main__":

    print("*************-------------------*******************")
    cpucode()
    cpucodedual()
    basecode()
    basecodedual()
