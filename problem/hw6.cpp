#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>
#include <math.h>
#include <map> 
#include <tuple>
#include <random>
#include <ctime>
#include <algorithm>

using namespace std;

double sigmoid(double x){
    return 1/(1+ exp(-1*x));
};

class neuron{

    public:
        int id;
        list<int> par_list;
        list<int> child_list;
        
        double net;
        double output;
        bool hasConstant = false;
        
        double delta;
        double gradient;
    
    neuron(int id){
        this->id = id;
        this->net = 0;
        this->output = 0;
        this->gradient = 0;
        
    };
    
    ~neuron(){
    };
    
    void addPar(int id){
        par_list.push_back(id);
    };
    
    void addChild(int id){
        child_list.push_back(id);
    };
};


class neural_network{
    
    public:
        map<int, neuron*> node_dict;
        map<tuple<int,int>, double> edge_dict;

        map<tuple<int,int>, double> edge_gradient_dict;
        
        vector<vector<double>> in_data;
        vector<vector<double>> out_data;
        
        list<int> topologyOrder;
        
        double learn_rate = 0.1;
        
        int batch_size = 100;
        
        default_random_engine generator;
        normal_distribution<double> distribution;
        
    neural_network(){
        this->setInAndOutlayer();
        
    };

    ~neural_network(){
    }
    
    vector<double> predict(vector<double> inputData){
        
        list<int>::iterator it;
        for(it = this->topologyOrder.begin(); it != this->topologyOrder.end(); ++it){
            int node_id = (*it);
            neuron* node = this->node_dict[node_id];
            
            // Input node
            if( node_id >= 0 && node_id < 27 ){
                node->output = inputData[node_id];
            }
            else if( node_id < 36 ){ // output node
                double net = 0;
                
                list<int>::iterator par_it;
                for( par_it = node->par_list.begin(); par_it != node->par_list.end(); ++par_it){
                    int par_id = (*par_it);
                    net += this->node_dict[par_id]->output * this->edge_dict[make_tuple(par_id, node_id)];
                }
                if( node->hasConstant ){
                    net += this->edge_dict[make_tuple(node_id, node_id)];
                }
                node->net = net;
            }
            else{
                double net = 0;
                
                list<int>::iterator par_it;
                for( par_it = node->par_list.begin(); par_it != node->par_list.end(); ++par_it){
                    int par_id = (*par_it);
                    net += this->node_dict[par_id]->output * this->edge_dict[make_tuple(par_id, node_id)];
                }
                if( node->hasConstant ){
                    net += this->edge_dict[make_tuple(node_id, node_id)];
                }
                
                node->net = net;
                node->output = sigmoid(net);
            }
            
        }
        
        vector<double> prediction;
        // Compute Softmax
        double sigma = 0; // Summation of net output
        for(int i = 27; i <= 35; i++){
            sigma += exp(this->node_dict[i]->net + pow(10,-6));
        }
        for(int i = 27; i <= 35; i++){
            
            neuron* node = this->node_dict[i];
            node->output = exp( node->net  + pow(10,-6) ) / sigma;
            prediction.push_back(node->output);
        }
        
        return prediction;
    }
    
    void loadData(string infilename, string truefilename){
        
        cout << "Start Loading Data" << endl;
        string line;
        fstream myfile;
        
        int count1 = 0;
        int count2 = 0;
        // Read infile
        myfile.open(infilename, ios::in);
        while( getline(myfile, line)){
            istringstream iss(line);
            double num;
            vector<double> v;  
            while(iss >> num){
                v.push_back(num);
            }
            this->in_data.push_back(v);
            count1++;
        }
        myfile.close();
        
        // Read truefile
        myfile.open(truefilename, ios::in);
        while( getline(myfile, line)){
            istringstream iss(line);
            double num;
            vector<double> v;  
            while(iss >> num){
                v.push_back(num);
            }
            this->out_data.push_back(v);
            count2++;
        }
        myfile.close();
        
        cout << "Data loaded" << endl;
        
        cout << "input file line : " << count1 << "; output file line : " << count2 << endl;
    }
   
    void setInAndOutlayer(){
        for(int id = 0; id < 36; id++){
            this->addNode(id);
        }
    
    }
    
    void readTopology(string filename){
        string line;
        fstream myfile;
        myfile.open(filename,ios::in);
        
        int node_count = 0;
        int line_count = 0;
        while (getline(myfile, line))
        {
            std::istringstream iss(line);
            int node_id, edge_id1, edge_id2;
            double weight;
            std::vector<double> v;

            if( line_count == 0 ){
                int node_id;
                while(iss >> node_id){
                    node_count++;
                    this->addNode(node_id);
                }
            }else{
                
                iss >> edge_id1;
                iss >> edge_id2;
                
                if(iss >> weight){
                    this->addWeight(make_tuple(edge_id1, edge_id2),weight);
                }else{
                    this->addWeight(make_tuple(edge_id1, edge_id2));
                }
            
            }
            
            line_count++;
          
            //for ( std::vector<double>::const_iterator i = v.begin(); i != v.end(); ++i)
            //    std::cout << *i << ' ';

        }
        
        myfile.close();
        
        cout << "Node count = " << node_count << endl;
        cout << "Edge count = " << line_count -1 << endl;
    
    
        this->topologicalSort();
    }
    
    void topologicalSort(){
            
            cout << "TopologicalSort Start" << endl;
            list<int> L, S;
            map<int, int> parCountDict;
            
            map<int, neuron*>::iterator it;
            for ( it = this->node_dict.begin(); it != this->node_dict.end(); it++ )
            {
                if( it->second->par_list.size() == 0){
                    S.push_back(it->first);
                }
                else{
                    parCountDict[it->first] = it->second->par_list.size();
                }
            }

            while( S.size() != 0){
                int par_id = S.front();
                S.pop_front();
                L.push_back(par_id);
                
                list<int> child_list = this->node_dict[par_id]->child_list;
                list<int>::iterator it_list;
                for (it_list = child_list.begin(); it_list != child_list.end(); ++it_list){
                    int child_id = *it_list;
                    parCountDict[child_id] -= 1;
                    if( parCountDict[child_id] == 0){
                        S.push_back(child_id);
                        parCountDict.erase(child_id); 
                    }
                }
            }
            
            this->topologyOrder = L;
            
            if( parCountDict.size() != 0 )
                cout << "Network is not a DAG" << endl;
            cout << "TopologicalSort Done" << endl;
            this->printTopologyOrder();
    }
    
    void printTopologyOrder(){
    
        list<int>::iterator it;
        
        cout << "TopologyOrder : ";
        for(it = this->topologyOrder.begin(); it != this->topologyOrder.end(); ++it){
            cout << (*it) << " ";
        }
        cout << endl;

    }
    
    void addWeight(tuple<int,int> edge_pair){
        
        
        static default_random_engine generator(time(0));
        static normal_distribution<double> distribution(0.0,1.0);
        
        this->edge_dict[edge_pair] = distribution(generator); // weight should be initialized randomly
        this->edge_gradient_dict[edge_pair] = 0;
        
        if( get<0>(edge_pair) != get<1>(edge_pair) ){
            // par_list & child_list
            int par_id = get<0>(edge_pair);
            int child_id = get<1>(edge_pair);
            neuron* par_node = this->node_dict[par_id];
            neuron* child_node = this->node_dict[child_id];
            
            par_node->addChild(child_id);
            child_node->addPar(par_id);
        }else{
            this->node_dict[get<0>(edge_pair)]->hasConstant = true;
        }
        
    };
    
    void addWeight(tuple<int,int> edge_pair, double weight){
        
        this->edge_dict[edge_pair] = weight;
        this->edge_gradient_dict[edge_pair] = 0;
        
        if( get<0>(edge_pair) != get<1>(edge_pair) ){
            
            // par_list & child_list
            int par_id = get<0>(edge_pair);
            int child_id = get<1>(edge_pair);
            neuron* par_node = this->node_dict[par_id];
            neuron* child_node = this->node_dict[child_id];
            par_node->addChild(child_id);
            child_node->addPar(par_id);
        }else{
            this->node_dict[get<0>(edge_pair)]->hasConstant = true;
        }
    };
    
    void addNode(int id){
        neuron* node = new neuron(id);
        this->node_dict[id] = node; 
    };

    void backPropagation(vector<double> input_x, vector<double> target){
        
        // To compute y - t
        vector<double> prediction = this->predict(input_x); // prediction

        vector<double> delta_vec;
        for(int i = 0; i < 9; i++){
            //cout << prediction[i];
            delta_vec.push_back( prediction[i] - target[i]);
        }
        //cout << endl;
        neuron* node;
        // Reverse order
        for (list<int>::reverse_iterator rit=this->topologyOrder.rbegin(); rit!=this->topologyOrder.rend(); ++rit){
            node = this->node_dict[*rit];
            int node_id = node->id;
            // Output_node
            if(node_id > 26 && node_id <= 35){
                node->delta = delta_vec[node_id-27];
                    //cout << node->delta << endl;
                for (list<int>::iterator it=node->par_list.begin(); it!=node->par_list.end(); ++it){
                    neuron* par_node = this->node_dict[(*it)];
                    int par_id = (*it);
                    this->edge_gradient_dict[make_tuple(par_id,node_id)] += node->delta * par_node->output ;
                }
                
                if(node->hasConstant){
                    this->edge_gradient_dict[make_tuple(node_id,node_id)] += node->delta ;
                }
            }else if( node_id > 35 ){// Non Input_node
                
                // Compute delta
                double delta = 0;
                for (list<int>::iterator it=node->child_list.begin(); it!=node->child_list.end(); ++it){
                    
                    neuron* child_node = this->node_dict[(*it)];
                    int child_id = (*it);
                    
                    delta += child_node->delta * this->edge_dict[make_tuple(node_id, child_id)];
                    
                }
                node->delta = delta * sigmoid(node->net) * (1 - sigmoid(node->net) );                
                // Backprogation to parents
                
                for (list<int>::iterator it=node->par_list.begin(); it!=node->par_list.end(); ++it){
                    
                    neuron* par_node = this->node_dict[(*it)];
                    int par_id = (*it);
                    this->edge_gradient_dict[make_tuple(par_id,node_id)] += node->delta * par_node->output ;
                    //cout << "gradient : "<< this->edge_gradient_dict[make_tuple(par_id,node_id)] << endl;
                }
                
                // Update constant weight
                if(node->hasConstant){
                    this->edge_gradient_dict[make_tuple(node_id,node_id)] += node->delta ;
                }
            }
            // 
        }
    };
        
    void update_edge(double size){
        map<tuple<int,int>, double>::iterator it;
        for ( it = this->edge_dict.begin(); it != this->edge_dict.end(); it++ ){
            this->edge_dict[it->first] -= this->learn_rate * this->edge_gradient_dict[it->first] / size;
            this->edge_gradient_dict[it->first] = 0;
        }
    };

    void trains(double learn_rate, double batch_size){
        
        cout << "Start training" << endl;
        this->learn_rate = learn_rate;
        
        int counter = 0;

        for (vector<vector<double>>::iterator it = this->in_data.begin() ; it != this->in_data.end(); ++it){
            
            this->backPropagation( (*it), this->out_data[counter]);
            if( counter % int(batch_size) == 0){
                this->update_edge( double(batch_size) );
            }
            
           counter++;
            
        }
        
        if( double((this->in_data.size())%(int(batch_size))) != 0 )
            this->update_edge( double((this->in_data.size())%(int(batch_size))) );
        //cout << "End training" << endl;
    };
    
    void compute_loss(){
        
        int counter = 0;
        double loss = 0;
        for (vector<vector<double>>::iterator it = this->in_data.begin() ; it != this->in_data.end(); ++it){
            
            vector<double> prediction = this->predict((*it));
            vector<double> target = this->out_data[counter];
            
            int inner_counter = 0;
            for( vector<double>::iterator it2 = prediction.begin(); it2 != prediction.end(); ++it2){
                loss += (-1)* target[inner_counter] * log( (*it2) + pow(10.0, -6));
                inner_counter++;
            }
            
            counter++;
        }
        
        cout << "Loss = " << loss/double(this->in_data.size())<< endl;
    }

    void writeStructureAndWeight(string output_filename){
        
        fstream myfile;    
        myfile.open(output_filename, ios::out);
        if (myfile.is_open()){
            for( map<int,neuron*>::iterator it = this->node_dict.begin(); it != this->node_dict.end(); it++ ){
                if( (it->first) > 35)
                    myfile << (it->first) << " ";
            }
            myfile << "\n";
            
            for( map<tuple<int,int>,double>::iterator it = this->edge_dict.begin(); it != this->edge_dict.end(); it++ ){
                myfile << get<0>(it->first) << " " << get<1>(it->first) << " " << it->second << "\n";
            }
            myfile.close();
        }
        
    
    };

    void testWithRandomAgent(){
    
        int winCount = 0;
        int tieCount = 0;
        
        //Test for 10000 times
        for(int i = 0; i<10000; i++){
            
            vector<double> table;
            // Initial table
            for(int i = 0; i<9; i++){
                table.push_back(0);
            }
            vector<double> transform;
            for(int i =0; i<27; i++){
                if (i%3 == 0){
                  transform.push_back(1);
                }
                else{
                transform.push_back(0);
                }
            }
            while(true){
                    

                // User Fisrt
                vector<double> prediction = this->predict(transform);
                for( vector<double>::iterator it = prediction.begin(); it != prediction.end(); it++ ){
                    int index = it - prediction.begin();
                    if( table[index] != 0 ){
                        prediction[index] = 0;
                    }
                }

                int argMax = distance(prediction.begin(), max_element(prediction.begin(), prediction.end()));

                table[argMax] = 1;
                transform[3*argMax] = 0;
                transform[3*argMax+1] = 1;
                
                // Check Game is ended?
                if( this->whoWin(table) != 0 ){
                    if( whoWin(table) == 1){
                        //cout << "Round " << i <<": User Win" << endl; 
                        winCount++;
                    }else if( whoWin(table) == -1){
                        //cout << "Round " << i <<": RandomAgent Win" << endl;                      
                    }
                    break;
                }
                if( this->isGameEnd(table) ){
                    tieCount++;
                    break;
                }
                
                // RandomAgent Second
                int ind = this->randomAgent(table);
                table[ind] = -1;
                transform[3*ind] = 0;
                transform[3*ind+2] = 1;
 

                // Check
                if( this->whoWin(table) != 0 ){
                    if( whoWin(table) == 1){
                        //cout << "Round " << i <<": User Win" << endl; 
                        winCount++;
                    }else if( whoWin(table) == -1){
                        //cout << "Round " << i <<": RandomAgent Win" << endl; 
                        
                    }
                    break;
                }
                if( this->isGameEnd(table) ){
                    tieCount++;
                    break;
                }

            }
        }
        
        cout << "WinCount = " << winCount << endl;
        cout << "TieCount = " << tieCount << endl;
        cout << "LossCount = " << 10000 - winCount - tieCount << endl;
    };
    
    //vector<double> randomAgent( vector<double> table){
    int randomAgent( vector<double> table){
        
        static default_random_engine generator_ra(time(0));
        static uniform_int_distribution<int> distribution_ra(1,10000);
        
        
        vector<int> availableNodeVec;
        
        for( vector<double>::iterator it = table.begin(); it != table.end(); it++ ){
            if( (*it) == 0){
                availableNodeVec.push_back(it - table.begin());
            }
        }
        
        int RandIndex = distribution_ra(generator_ra) % availableNodeVec.size();
        return availableNodeVec[RandIndex]; 
        //vector<double> returnVec;
        //returnVec.assign(table.begin(), table.end());
        //returnVec[availableNodeVec[RandIndex]] = -1;
        
        //return returnVec;
        
    };
    
    int whoWin( vector<double> table){
    
        for(int i = 0 ; i<3; i++){
            int j = i * 3;
            if( table[j] == table[j+1] && table[j+1] == table[j+2] && table[j+2] != 0){
                return table[j];
            }
            if( table[i] == table[i+3] && table[i+3] == table[i+6] && table[i+6] != 0){
                return table[i];
            }
        }
        if( table[0] == table[4] && table[4] == table[8] && table[8] != 0){
            return table[0];
        }
        if( table[2] == table[4] && table[4] == table[6] && table[6] != 0){
            return table[2];
        }
        return 0;
        
    };
    
    

    void printVector(vector<double> pre){
        
        for( vector<double>::iterator it = pre.begin(); it!=pre.end(); it++){
            cout << (*it) << " ";
            if( (it - pre.begin()) % 3 == 2){
                cout << endl ;
            }
        }
        cout << endl;
    };

    bool isGameEnd( vector<double> table){
        
        for( vector<double>::iterator it = table.begin(); it != table.end(); it++){
            if( *it == 0 ){
                return false;
            }
        }
        return true;
    };
    
};


void printVector(vector<double> pre){
    
    for( vector<double>::iterator it = pre.begin(); it!=pre.end(); it++){
        cout << (*it) << " "<< endl;
        if( (it - pre.begin()) % 3 == 2){
            cout << endl ;
        }
    }
    cout << endl;
};


int main(int argc, char* argv[]){

    double learn_rate = atof(argv[1]);
    int epochs = atoi(argv[2]);
    double batch_size = atof(argv[3]);
    string model_name = argv[4];
    string input_x_filename = argv[5];
    string input_y_filename = argv[6];
    string output_filename = argv[7];
    int isTest = atoi(argv[8]);
    
    neural_network* nn = new neural_network();
    nn->readTopology(model_name);
    nn->loadData(input_x_filename,input_y_filename);

    for(int i = 0; i < epochs; i++){
    	cout << "Training start" << endl;
    	cout << "epochs : " << i+1 << endl;
        nn->trains(learn_rate, batch_size);
        
        nn->compute_loss();
        
    }
    nn->writeStructureAndWeight(output_filename);
    if( isTest != 0)
        nn->testWithRandomAgent();
    
    
    return 0;
};
