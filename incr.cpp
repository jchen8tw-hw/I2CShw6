#include<fstream>
#include<iostream>
using namespace std;
int main(){
	ofstream out("model_new.txt");
	for(int i = 36;i<36+18*3;i++){
		if(i == 36+18*3-1) out << i << endl;
		else out << i <<" ";
	}
	for(int i = 36;i<36+18;i++){
		for(int j = 36+18;j<36+18+18;j++){
			out << i << " "<< j << endl;
		}
	}
	for(int i = 36+18;i<36+18+18;i++){
		for(int j = 36+18+18;j<36+18*3;j++){
			out << i << " " << j << endl;
		}
	}
	for(int i = 36+18+18;i<36+18*3;i++){
		for(int j = 27;j<36;j++){
			out << i << " " << j << endl;
		}
	}
	for(int i = 36;i<36+18*3;i++){
		out << i << " " << i << endl;
	}
	out.close();
	return 0;
}