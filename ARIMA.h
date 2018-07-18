#pragma once
#include <algorithm>
#include "iostream"
#include "math.h"
#include "stdlib.h"
#include "vector"

#include "assert.h"

using namespace std;


#define Type double
#define InType int
#define Datasize 15
#define P 1
#define D 1
#define Q 1
class DataServer
{
public:
	DataServer()
	{
		aveValue = 0;
	}
	vector<vector<Type>> get_Cooked_value(vector<vector<InType >>& InputData)
	{
		Normalization(InputData);

		return cookedData;
	}

	Type aveValue;
	vector<vector<Type>> cookedData;
	vector<Type> lastWeek;
private:
	inline void Normalization(vector<vector<InType >>& InputData)
	{
		cookedData.clear();
		GetAavevalue(InputData);
		if (aveValue <= 0)
		{
			return;
		}
		vector<Type> temp;		
		for (int i = 0; i < InputData.size(); i++)
		{
			temp.clear();
			for (int j = 0; j < InputData[i].size(); j++)
			{
				temp.push_back((Type)InputData[i][j]- aveValue);
			}
			cookedData.push_back(temp);
		}
		get_LastWeek();
	}
	inline void GetAavevalue(vector<vector<InType >>& InputData)
	{
		int cnt = 0;
		for (int i = 0; i < InputData.size(); i++)
		{
			for (int j = 0; j < InputData[i].size(); j++)
			{
				aveValue +=InputData[i][j];
				cnt++;
			}
		}
		aveValue = aveValue / cnt;
	}
	inline void get_LastWeek()
	{
		lastWeek.clear();
		vector<Type>::iterator it;
		for (int i = 0; i < cookedData.size(); i++)
		{
			it = cookedData[i].end() - 1;
			lastWeek.push_back(*it);
			it=cookedData[i].erase(it);
		}
	}
};

class arimaModel
{
public:
	arimaModel()
	{	
		cur_p = 0;
		cur_q=0;

		initializeW((Type*)AR, Datasize*P);
		initializeW((Type*)MA, Datasize*Q);
		MA0 = initializeW();
		AR0 = initializeW();
	}
	~arimaModel()
	{

	}

	int runingArima(vector<vector<InType>>& input)
	{
		Dataprepare(input);
		int errorflag = 0;
		for (int i = 0; i < Normed_Data.size()&&errorflag==0; i++)
		{
			errorflag = oneFullTurn(i);			
		}
		return errorflag;
	}	
	int oneFullTurn(int idx,int level=1)
	{		
		if (!GetytP())
			return -3;
		if (!Getyt(idx,level))
			return -2;
		if (!Getut())
			return -1;
		return 0;
	}
protected:
	inline Type initializeW()
	{
		return ((Type)(2.0 * rand()) / ((Type)RAND_MAX + 1.0) - 1.0);
	}
	inline void initializeW(Type w[], int num)
	{
		
		for (int i = 0; i < num; i++)
		{
			w[i] = ((Type)(2.0 * rand()) / ((Type)RAND_MAX + 1.0) - 1.0);
		}
	}
	inline void Dataprepare(vector<vector<InType>>& input)
	{
		DataServer* server = new(DataServer);
		vector<vector<Type>> temp;

		(server->get_Cooked_value(input)).swap(temp);
		(server->lastWeek).swap(NormedLastWeek);

		Normed_Data.clear();
		NormedLastWeek.clear();
		vector<Type> temp1;
		for (int i = 0; i < temp.size();i++)
		{
			temp1.clear();
			for (int j = 0; j <Datasize; j++)
			{
				temp1.push_back(temp[j][i]);
			}
			Normed_Data.push_back(temp1);
		}
	}
	inline bool Getyt(int idx, int level)
	{
		if (Normed_Data.size() <= idx || Normed_Data[idx].size() != Datasize)
		{
			return false;
		}

		cur_yt.clear();
		for (int i = 0; i < Datasize; i++)
		{
			if (level == 1)
			{
				if (idx < 1)
					cur_yt.push_back(Normed_Data[idx][i]);
				else
					cur_yt.push_back(Normed_Data[idx][i] - Normed_Data[idx - 1][i]);
			}
			else if (level == 2)
			{
				if (idx >= 2)
				{
					cur_yt.push_back(Normed_Data[idx][i] - 2 * Normed_Data[idx - 1][i] + Normed_Data[idx - 2][i]);
				}
				else if (idx == 0)
				{
					cur_yt.push_back(Normed_Data[idx][i]);
				}
				else
				{
					cur_yt.push_back(Normed_Data[idx][i] - 2 * Normed_Data[idx - 1][i]);
				}

			}

		}
		if (cur_yt.size() != Datasize)
			return false;
		return true;
	}
	inline bool GetytP()
	{
		cur_ytP.clear();
		Type* ar = new(Type);
		Type* ma = new(Type);

		for (int i = 0; i < Datasize; i++)
		{
			cur_ytP.push_back( AR0 + MA0);

			int j = 0;
			ar = AR[i];
			ma = MA[i];
			int mini = min(cur_p, cur_q);
			for (j = 0; j < mini; j++)
			{
				cur_ytP[i] += ar[i*P + j] * yts[j][i] + ma[i*Q + j] * uts[j][i];
			}
			if (cur_p >= cur_q)
			{
				for (j = mini; j < cur_p; j++)
				{
					cur_ytP[i] += ar[i*P + j] * yts[j][i];
				}
			}
			else
			{
				for (j = mini; j <cur_q; j++)
				{
					cur_ytP[i] += ma[i*Q + j] * uts[j][i];
				}
			}
		}

		ar = NULL;
		ma = NULL;
		delete[] ar;
		delete[] ma;
		if (cur_ytP.size() != Datasize)
			return false;

		UpDate_yt();
		return true;
	}
	inline bool Getut()
	{
		if (cur_yt.size() != cur_ytP.size())
		{
			return false;
		}
		cur_ut.clear();
		for (int i = 0; i < Datasize; i++)
		{
			cur_ut.push_back(cur_yt[i] - cur_ytP[i]);
		}
		UpDate_ut();
		return true;
	}
	inline void UpDate_yt()
	{
		if (yts.size() < P)
		{
			yts.push_back(cur_ytP);
			cur_p++;
			return;
		}
		vector<vector<Type>> temp1;
		temp1.insert(temp1.end(), yts.begin() + 1, yts.end());
		temp1.push_back(cur_ytP);
		yts.clear();
		temp1.swap(yts);
	}
	inline void UpDate_ut()
	{
		if (uts.size() < Q)
		{
			uts.push_back(cur_ut);
			cur_q++;
			return;
		}
		vector<vector<Type>> temp2;
		temp2.insert(temp2.end(), uts.begin() + 1, uts.end());
		temp2.push_back(cur_ut);
		uts.clear();
		temp2.swap(uts);
	}
		
private:
	vector<vector<Type>> Normed_Data;
	vector<Type> NormedLastWeek;

	int cur_p, cur_q;
	Type AR[Datasize][P];
	Type MA[Datasize][Q];	
	Type AR0, MA0;

	vector<Type> cur_yt;;
	vector<Type> cur_ytP;
	vector<Type> cur_ut;
	vector<vector<Type>> yts;
	vector<vector<Type>> uts;
};
