#include <iostream>
#include <fstream>
#include "./include/ann.h"
#include <vector>

int main(){
    ANN test(4,1,10,3);
    
    std::vector<std::string> af; af.resize(1); af = {"sigmoid", "sigmoid"};
    test.set_activation_functions(af);

    //neural netowrk params
    test.set_learning_rate(0.05);
    test.set_momentum(0.05);
    test.set_desired_error(0.0001);
    test.set_max_epochs(1000);
    std::ifstream trainfile("iris_train.txt");
    std::ifstream testfile("iris_test.txt");

    int train_count, num_in, num_out;
    trainfile >> train_count >> num_in >> num_out;
    
    std::vector< std::vector<double> > dataset;
    std::vector< std::vector<double> > expected_outputs;

    dataset.resize(train_count);
    expected_outputs.resize(train_count);

    for(int i = 0; i < train_count; i++){
        dataset[i].resize(num_in);
        trainfile >> dataset[i][0] >> dataset[i][1] >> dataset[i][2] >> dataset[i][3];
        expected_outputs[i].resize(num_out);
        trainfile >> expected_outputs[i][0] >> expected_outputs[i][1] >> expected_outputs[i][2];    
    }

   test.batch_learn(dataset, expected_outputs, train_count);

    std::vector< std::vector<double> > test_data;
    std::vector< std::vector<double> > prediction;
    std::vector< std::vector<int> > correct_outputs;
    int testcount, testin, testout;
    testfile >> testcount >> testin >> testout;
    
    test_data.resize(testcount);
    prediction.resize(testcount);
    correct_outputs.resize(testcount);
    
    for(int i = 0; i < testcount; i++){
        test_data[i].resize(testin);
        testfile >> test_data[i][0] >> test_data[i][1] >> test_data[i][2] >> test_data[i][3];
        correct_outputs[i].resize(testout);
        testfile >> correct_outputs[i][0] >> correct_outputs[i][1] >> correct_outputs[i][2];
        
        test.run(test_data[i]);
        prediction[i] = test.return_output();

    }
    
    for(int i = 0; i < testcount; i++){
        std::cout << "----TEST " << i << " ----" << "\n";
        std::cout << "INPUT DATA: ";
        for(int j = 0; j < test_data[i].size(); j++){
            std::cout << test_data[i][j] << " ";
        }
        std::cout << "\n";
        std::cout << "PREDICTIONS: ";
        for(int j = 0; j < prediction[i].size(); j++){
            std::cout << prediction[i][j] << " ";
        }
        std::cout << "\n";
        std::cout << "EXPECTED OUTPUT: ";
        for(int j = 0; j < correct_outputs[i].size(); j++){
            std::cout << correct_outputs[i][j] << " ";
        }
        std::cout << "\n\n";
    }


    return 0;
}
