#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"

#define HOG_SIZE 20
#define H_BLOCKS 8
#define W_BLOCKS 8

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

// Exatract features from dataset.
// You should implement this function by yourself =)
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
		BMP *tmp_img = data_set[image_idx].first;
        int h = tmp_img->TellHeight();
        int w = tmp_img->TellWidth();

        Matrix<float> grayscale(h, w);
        for (int i = 0; i < h; ++i) {
        	for (int j = 0; j < w; ++j) {
        		int r, g, b;
        		r = (tmp_img->GetPixel(j, i)).Red;
        		g = (tmp_img->GetPixel(j, i)).Green;
        		b = (tmp_img->GetPixel(j, i)).Blue;
        		grayscale(i, j) = 0.299 * r + 0.587 * g + 0.114 * b;
        	}
        }

        Matrix<float> sobel_x(h, w);
        Matrix<float> sobel_y(h, w);
        Matrix<float> direction(h, w);
        Matrix<float> module(h, w);
        for (int i = 0; i < h; ++i) {
        	for (int j = 0; j < w; ++j) {
        		if ((j == 0) || (j == w - 1)) {
        			sobel_x(i, j) = grayscale(i, j);
        		} else {
        			sobel_x(i, j) = -grayscale(i, j - 1) + grayscale(i, j + 1);
        		}
        		if ((i == 0) || (i == h - 1)) {
        			sobel_y(i, j) = grayscale(i, j);
        		} else {
        			sobel_y(i, j) = grayscale(i - 1, j) - grayscale(i + 1, j);
        		}
        		module(i, j) = sqrt(sobel_x(i, j) * sobel_x(i, j) + sobel_y(i, j) * sobel_y(i, j));
        		direction(i, j) = atan2(sobel_y(i, j), sobel_x(i, j)) + M_PI;
        	}
        }
	    
        vector<float> hog;
        int h_block_size = h / H_BLOCKS;
        int w_block_size = w / W_BLOCKS;
        
        int ind = 0;
        for (int h_b = 0; h_b < H_BLOCKS; ++h_b) {
        	for (int w_b = 0; w_b < W_BLOCKS; ++w_b) {
        		vector<float> cur_hog(HOG_SIZE, 0);
        		float sum = 0;
        		for (int dh = 0; dh < h_block_size; ++dh) {
        			for (int dw = 0; dw < w_block_size; ++dw) {
        				ind = direction(h_b * h_block_size + dh, w_b * w_block_size + dw) / (2 * M_PI / HOG_SIZE);
        				cur_hog[ind] += module(h_b * h_block_size + dh, w_b * w_block_size + dw);
        				sum += pow(module(h_b * h_block_size + dh, w_b * w_block_size + dw), 2);
        			} 
        		}
        		if (sum > 1e-6) {
        			for (int i = 0; i < HOG_SIZE; ++i) {	
        				cur_hog[i] = cur_hog[i] / sqrt(sum);
        			}
        		}
        		for (int i = 0; i < HOG_SIZE; ++i){
        			hog.push_back(cur_hog[i]);	
        		}
        	}
        }

        //color signs (additional task #2)
        vector<float> color_signs;
        for (int h_b = 0; h_b < H_BLOCKS; ++h_b) {
            for (int w_b = 0; w_b < W_BLOCKS; ++w_b) {
                float r_avg = 0;
                float g_avg = 0; 
                float b_avg = 0;
                for (int dh = 0; dh < h_block_size; ++dh) {
                    for (int dw = 0; dw < w_block_size; ++dw) {
                        r_avg = r_avg + (tmp_img->GetPixel(w_b * w_block_size + dw, h_b * h_block_size + dh)).Red;
                        g_avg = g_avg + (tmp_img->GetPixel(w_b * w_block_size + dw, h_b * h_block_size + dh)).Green;
                        b_avg = b_avg + (tmp_img->GetPixel(w_b * w_block_size + dw, h_b * h_block_size + dh)).Blue;
                    }
                }
                r_avg = r_avg / (h_block_size * w_block_size);
                g_avg = g_avg / (h_block_size * w_block_size);
                b_avg = b_avg / (h_block_size * w_block_size); 
                color_signs.push_back(r_avg / 255.0);
                color_signs.push_back(g_avg / 255.0);
                color_signs.push_back(b_avg / 255.0);               
            }
        }
        for (size_t i = 0; i < color_signs.size(); ++i) {
            hog.push_back(color_signs[i]);
        }

        //LBP (additional task #1)
        vector<float> lbp;
        h_block_size = (h - 2) / H_BLOCKS;
        w_block_size = (w - 2) / W_BLOCKS;
        int cur_num, cur_i, cur_j;
        for (int h_b = 0; h_b < H_BLOCKS; ++h_b) {
            for (int w_b = 0; w_b < W_BLOCKS; ++w_b) {
                vector<float> cur_lbp(256, 0);
                for (int dh = 0; dh < h_block_size; ++dh) {
                    for (int dw = 0; dw < w_block_size; ++dw) {
                        cur_i = h_b * h_block_size + dh + 1;
                        cur_j = w_b * w_block_size + dw + 1;
                        cur_num = 0;
                        if (grayscale(cur_i, cur_j) <= grayscale(cur_i - 1, cur_j - 1)) {
                            cur_num += 128;
                        }
                        if (grayscale(cur_i, cur_j) <= grayscale(cur_i - 1, cur_j)) {
                            cur_num += 64;
                        }
                        if (grayscale(cur_i, cur_j) <= grayscale(cur_i - 1, cur_j + 1)) {
                            cur_num += 32;
                        }
                        if (grayscale(cur_i, cur_j) <= grayscale(cur_i, cur_j + 1)) {
                            cur_num += 16;
                        }
                        if (grayscale(cur_i, cur_j) <= grayscale(cur_i + 1, cur_j + 1)) {
                            cur_num += 8;
                        }
                        if (grayscale(cur_i, cur_j) <= grayscale(cur_i + 1, cur_j)) {
                            cur_num += 4;
                        }
                        if (grayscale(cur_i, cur_j) <= grayscale(cur_i + 1, cur_j - 1)) {
                            cur_num += 2;
                        }
                        if (grayscale(cur_i, cur_j) <= grayscale(cur_i, cur_j - 1)) {
                            cur_num += 1;
                        }
                        cur_lbp[cur_num] += 1;
                    }
                }
                for (size_t i = 0; i < cur_lbp.size(); ++i) {
                    lbp.push_back(cur_lbp[i] / (h_block_size * w_block_size));
                }
            }
        }
        for(size_t i = 0; i < lbp.size(); ++i) {
            hog.push_back(lbp[i]);
        }

        features->push_back(make_pair(hog, data_set[image_idx].second));
    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}