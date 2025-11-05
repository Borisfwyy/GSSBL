from SIF import run_sif
from uSIF import run_usif

from SIFsvm import run_sif_svm
from uSIFsvm import run_usif_svm


from BoW import run_bow
from TextCNN import run_textcnn
from te import run_bert
from bilstm import run_siamese
from GSSBL import run_weighted_fusion_siamese
from canshu import run_weighted_fusion_siamese_canshu
from jvtrans import run_weighted_fusion_siamese_trans
from savemodel import save_weighted_fusion_siamese
from saveresult import saveresult_weighted_fusion_siamese


traindata = "dataset/1:10/train.txt"
testdata = "dataset/1:10/test.txt"

result_path = "result/yuzhizhu/"

wordvec = "model/sg512main.json" #cbow512main.json sg
wordvecsub = "model/sg512sub.json" #cbow512sub.json
wordvecmix = "model/sg1024combined.json" #cbow1024combined.json

wordvecshapevae = "model/glyph_vae_features.json" 
wordvecshapecon = "model/glyph_constractive.json" 
cuda = 4

def cpucode():
    
    run_sif(
        word_vec_path=wordvecsub,
        data_paths=[traindata, testdata],
        output_csv_path=result_path + "zi_sif.csv",
        repeat=5,
        cuda_device=cuda,
        use_subword=True
    )
    
    run_usif(
        word_vec_path=wordvecsub,
        data_paths=[traindata, testdata],
        output_csv_path=result_path + "zi_usif.csv",
        repeat=5,
        cuda_device=cuda,
        use_subword=True
    )
    
    run_bow(
        train_path=traindata,
        test_path=testdata,
        output_csv_path=result_path + "zi_bow.csv",
        repeat=5,
        cuda_device=cuda,
        use_subword=True #False使用主字,True使用子字
    )

def basecode():
    '''
    run_textcnn(
        train_path=traindata,
        test_path=testdata,
        word_vec_path=wordvec,
        output_csv_path=result_path + "zhu_textcnn.csv",
        repeat=5,
        cuda_device=cuda,
        use_subword=False
    )
    '''
    
    run_bert(
        train_path=traindata,
        test_path=testdata,
        word_vec_path=wordvec,
        output_csv=result_path + "zhu_transformer_lr1e-5.csv",
        repeat=5,
        cuda_device=cuda,
        use_subword=False
    )
    
    '''
    run_siamese(
        train_path=traindata,
        test_path=testdata,
        word_vec_path=wordvec,
        output_csv_path=result_path + "zhu_bilstm.csv",
        repeat=5,
        cuda_device=cuda,
        use_subword=False
    ) 
    '''
      
def transcode():
    # trans的句向量替代模型
    
    #主
    run_weighted_fusion_siamese_trans(
        train_path=traindata,
        test_path=testdata,
        word_vec_path_meaning= wordvec,
        word_vec_path_shape= wordvecshapevae,
        output_csv_path=result_path + "zhu_jvtrans_lr1e-6.csv",
        repeat=5,
        use_subword=False,
        cuda_device=cuda
    )
    '''
    #子
    run_weighted_fusion_siamese_trans(
        train_path=traindata,
        test_path=testdata,
        word_vec_path_meaning= wordvecsub,
        word_vec_path_shape= wordvecshapevae,
        output_csv_path=result_path + "zi_jvtransvae_lr1e-6.csv",
        repeat=5,
        use_subword=True,
        cuda_device=cuda
    )
    '''
    '''
    #对比
    run_weighted_fusion_siamese_trans(
        train_path=traindata,
        test_path=testdata,
        word_vec_path_meaning= wordvecsub,
        word_vec_path_shape= wordvecshapecon,
        output_csv_path=result_path + "zi_jvtranscon_lr1e-6.csv",
        repeat=5,
        use_subword=True,
        cuda_device=cuda
    )
    '''
          
def usecode():
    
    #主
    run_weighted_fusion_siamese(
        train_path=traindata,
        test_path=testdata,
        word_vec_path_meaning= wordvec,
        word_vec_path_shape= wordvecshapevae,
        output_csv_path=result_path + "glove_zhu_GSSBL.csv",
        repeat=5,
        use_subword=False,
        cuda_device=cuda
    )
    '''
    
    #子
    run_weighted_fusion_siamese(
        train_path=traindata,
        test_path=testdata,
        word_vec_path_meaning= wordvecsub,
        word_vec_path_shape= wordvecshapevae,
        output_csv_path=result_path + "glove_zi_GSSBLvae.csv",
        repeat=5,
        use_subword=True,
        cuda_device=cuda
    )
    
    
    #对比
    run_weighted_fusion_siamese(
        train_path=traindata,
        test_path=testdata,
        word_vec_path_meaning= wordvecsub,
        word_vec_path_shape= wordvecshapecon,
        output_csv_path=result_path + "glove_zi_GSSBLcon.csv",
        repeat=5,
        use_subword=True,
        cuda_device=cuda
    )  
    '''

if __name__ == "__main__":

    print("using:"+result_path)
    print("cuda:"+str(cuda))
    print("*************-------------------*******************")

    cpucode()
    basecode()
    usecode()
    transcode()


