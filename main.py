from GSSBL import run_weighted_fusion_siamese

traindata = "data/train.txt"
testdata = "data/test.txt"
result_path = "results/"
wordvec = "model/sgns.json"
wordvecshape = "model/glyph.json" 
use_subchara = True
cuda = 5


if __name__ == "__main__":
    run_weighted_fusion_siamese(
        train_path=traindata,
        test_path=testdata,
        word_vec_path_meaning= wordvec,
        word_vec_path_shape= wordvecshape,
        output_csv_path=result_path + "mean_results.csv",
        repeat=5,
        use_subword=use_subchara,
        cuda_device=cuda
    )
