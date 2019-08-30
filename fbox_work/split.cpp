#include<bits/stdc++.h>
#define FOR(i,n) for(int (i) = 0;(i) < (n);(i)++)
using namespace std;
const int maxn = 1e6 + 10;
const int INF = 1e9;
typedef long long ll;

inline int read(){
	int f = 1,x = 0;
	char ch = getchar();
	while(ch > '9' || ch < '0'){
		if(ch=='-') f = -1;ch = getchar();
	}
	while(ch <= '9' && ch >= '0'){
		x = x * 10 + ch - '0';		ch = getchar();
	}
	return f * x;
}

int main(){
//	ios::sync_with_stdio(false);
	ifstream fin("1.txt", std::ios::in);
	ofstream fout("2.txt");
	
	vector<string> vec[10];
	char line[1024];
	
	while(fin.getline(line, sizeof(line))){
		stringstream word(line);
		string str;
		for(int i = 1;i <= 6;i ++){
			word >> str;
			vec[i].push_back(str); 
		}
	} 
	
	for(int i = 1;i <= 6;i ++){
		for(auto j : vec[i]){
			fout << j << endl;
		}
	}
	
	return 0;
}
