#pragma once
#include <algorithm>
#include "iostream"
#include "math.h"
#include "stdlib.h"
#include "vector"
#include "assert.h"

using namespace std;
/**********************************************************Stiff&Hyper Params' List************************************************/
#define InType int
#define Type double
#define Datasize 15
#define Learning_step 7
#define Learning_rate 0.001
#define max_epoch 8000
#define Hidden 7 //when smaller,error sharper down and bounce,too big then fuck up,too small then big vibration in output
#define Gradient_tval 0.5
#define Error_limit 0.5
#define Layernum 2
bool MultiSet = true;
/***********************************************************Data Prepros Class****************************************************/

class DataServer
{
public:
	DataServer()
	{
		TopValue = 0;
	}
	vector<vector<Type>> get_Cooked_value(vector<vector<InType >>& InputData)
	{
		Normalization(InputData);
		return cookedData;
	}
	vector<Type> get_LastWeek()
	{
		lastWeek.clear();
		vector<Type>::iterator it;
		for (int i = 0; i < cookedData.size(); i++)
		{
			it = cookedData[i].end()-1 ;
			lastWeek.push_back(*it);
		}
		return lastWeek;
	}


	int TopValue;
	vector<vector<Type>> cookedData;
	vector<Type> lastWeek;
private:
	inline void Normalization(vector<vector<InType >>& InputData)
	{
		cookedData.clear();
		GetTopvalue(InputData);
		if (TopValue <= 0)
		{
			return;
		}
		vector<Type> temp;
		Type alpha = (Type)TopValue;
		for (int i = 0; i < InputData.size(); i++)
		{
			temp.clear();
			for (int j = 0; j < InputData[i].size(); j++)
			{
				temp.push_back((Type)InputData[i][j] / alpha);
			}
			cookedData.push_back(temp);
		}
	}
	inline void GetTopvalue(vector<vector<InType >>& InputData)
	{
		for (int i = 0; i < InputData.size(); i++)
		{
			for (int j = 0; j < InputData[i].size(); j++)
			{
				TopValue = max(InputData[i][j], TopValue);
			}
		}
	}
	
};


/*************************************************************GRUcell Class******************************************************/
class GRUcell
{
public:
	GRUcell(int firstOrLast,Type rate = Learning_rate, int learning_st = Learning_step, Type clip_gradient = Gradient_tval, int init = 1)
	{
		Global_count = 0;
		current_loss = 0;
		firstOrLast_flag = firstOrLast;
		/*loss_permit = error_permit;*/
		learning_step = learning_st;
		L_rate = rate;
		/*predict_l = predict_length;
		date_interval = pre_start_day;*/
		is_training = false;
		gradient_tval=clip_gradient;
		get_Gradient_weight();
		if (init == 1)
		{
			initAll();
		}
	}

	void initAll()
	{
		initializeW((Type*)wDoorZx, Hidden*Datasize);
		initializeW((Type*)wDoorRx, Hidden*Datasize);
		initializeW((Type*)wDoorZh, Hidden*Hidden);
		initializeW((Type*)wDoorRh, Hidden*Hidden);
		initializeW((Type*)wHPrh, Hidden*Hidden);
		initializeW((Type*)wHPx, Hidden*Datasize);
		initializeW((Type*)Wout, Hidden*Datasize);
	}
	/******************************************************************************************************************/
	void update_Hin()
	{
		Hinput.clear();
		if (Houtput.empty())
		{
			for (int i = 0; i < Hidden; i++)
			{
				Hinput.push_back(Type(0));
			}
		}
		else
		{
			Houtput.swap(Hinput);
		}
	}	
	int FrontPro(bool is_trainingFlag)
	{
		update_Hin();
		check_isTraining(is_trainingFlag);
		if (!FrontDoorUPDATE(outDoorR, outDoorRd, WrDatain, WrHin, (Type*)wDoorRx, (Type*)wDoorRh))
			return -6;
		if (!FrontDoorUPDATE(outDoorZ, outDoorZd, WzDatain, WzHin, (Type*)wDoorZx, (Type*)wDoorZh))
			return -5;
		if (!frontHpredict())
			return -4;
		if (!frontHoutput())
			return -3;
		if (!frontYoutput())
			return -2;
		if (firstOrLast_flag == -1&&is_training)
		{
			if (!get_L2_loss())
				return -1;
		}
		return 0;

	}
	int BackPro()
	{
		if (is_training)
		{
			if (GatherGradientW() != 0)
			{
				return -1;
			}
			step_count++;
			if (UpDateWeights())
			{
				step_count = 0;
				Gradient_reset();
			}
		}
		return 0;
	}
	void fillin_data(vector<Type>& input1, vector<Type>& input2)
	{
		Datainput.clear();
		input1.swap(Datainput);
		
		Targetinput.clear();
		input2.swap(Targetinput);		
	
	}
	void fillin_data(vector<Type>& input)
	{
		Datainput.clear();
		input.swap(Datainput);
	
	}

	/*Type decodeValue;*/
	vector<Type> Hinput;
	vector<Type> Datainput;
	vector<Type> Targetinput;
	vector<Type> Houtput;
	vector<Type> Yout;
	vector<Type> BackToLastCell;
	vector<Type> fromNextBack;

	int firstOrLast_flag;
	bool is_training;
	int learning_step;	
	int Global_count;
	int step_count;
	int predict_l;
	int date_interval;

	Type gradient_tval;
	Type current_loss;
	Type loss_permit;
	Type L_rate;
	/******************************************************************************************************************/
protected:

	/******************************************************************************************************************/
	//Gate(Layer) value
	vector<Type> outDoorZ, outDoorZd;
	vector<Type> outDoorR, outDoorRd;
	vector<Type> Hpredict, Hpredictd;
	vector<Type> Youtd;

	//middle param
	vector<Type> htAndrt;
	vector<Type> WrDatain, WrHin;
	vector<Type> WzDatain, WzHin;

	//weights
	Type wDoorZx[Hidden][Datasize];
	Type wDoorZh[Hidden][Hidden];
	Type wDoorRx[Hidden][Datasize];
	Type wDoorRh[Hidden][Hidden];
	Type wHPrh[Hidden][Hidden];
	Type wHPx[Hidden][Datasize];
	Type Wout[Datasize][Hidden];

	//About error and gradient
	Type Gradient_gather[6][Hidden];
	Type Gradient_gatherWo[Datasize];
	vector<Type> Gradient_weight;
	vector<Type> curr_er_ot;
	vector<Type> curr_hp_ht;
	vector<Type> Grad_Whx, Grad_Whrh;
	vector<Type> Grad_Wzx, Grad_Wzh;
	vector<Type> Grad_Wrx, Grad_Wrh;
	vector<Type> Grad_Wo;

	/******************************************************************************************************************/
private:
	//init
	inline void initializeW(Type w[], int num)
	{
		//Type* temp = new Type;
		for (int i = 0; i < num; i++)
		{		
			w[i] = ((Type)(2.0 * rand()) / ((Type)RAND_MAX + 1.0) - 1.0);
			
		}
	}
	inline void get_Gradient_weight()
	{
		Gradient_weight.clear();
		Type mom = (learning_step + 1)*learning_step / 2;
		for (int i = 0; i < learning_step; i++)
		{
			Gradient_weight.push_back((i + 1) *L_rate / mom);
		}
	}
	inline void check_isTraining(bool flag_training)
	{
		is_training = flag_training;
	}
	//basic util
	inline Type vectorSum(vector<Type>& input)
	{
		Type out = 0;
		for (int i = 0; i < input.size(); i++)
		{
			out += input[i];
		}
		return out;
	}	
	inline vector<Type> sigmoid(vector<Type>& input)
	{
		vector<Type> output;
		for (int i = 0; i < input.size(); i++)
		{
			output.push_back(1.0 / (1.0 + exp(-input[i])));
		}
		return output;
	}
	inline vector<Type> sigmoidd(vector<Type> & input)
	{
		vector<Type> output;
		Type temp;
		for (int i = 0; i < input.size(); i++)
		{			
			output.push_back(input[i] * (1.0 - input[i]));
		}
		return output;
	}
	inline vector<Type> tanh(vector<Type>& input)
	{
		vector<Type> output;
		for (int i = 0; i < input.size(); i++)
		{
			output.push_back((exp(input[i]) - exp(-input[i])) / (exp(input[i]) + exp(-input[i])));
		}
		return output;
	}
	inline vector<Type> tanhd(vector<Type>& input)
	{
		vector<Type> output;
		for (int i = 0; i < input.size(); i++)
		{
			output.push_back(1 - (input[i])*(input[i]));
		}
		return output;
	}
	inline vector<Type> Relu(vector<Type>& input)
	{
		vector<Type> out;
		for (int i = 0; i < input.size(); i++)
		{
			out.push_back(max(0., input[i]));
		}
		return out;
	}
	inline vector<Type> Relud(vector<Type>& input)
	{
		vector<Type> out;
		for (int i = 0; i < input.size(); i++)
		{
			if (input[i] > 0.)
			{
				out.push_back(0.);
			}
			else
			{
				out.push_back(1.);
			}
			
		}
		return out;
	}
	inline vector<Type> minusOrplus(vector<Type> input1, vector<Type> input2, int plusflag, int activeflag = 0, Type bias1 = 0, Type bias2 = 0)
	{
		vector<Type> out;
		if (input1.size() == input2.size())
		{
			if (plusflag == 0)
			{
				for (int i = 0; i < input1.size(); i++)
				{
					out.push_back(input1[i] + bias1 - input2[i] - bias2);
				}
			}
			if (plusflag == 1)
			{
				for (int i = 0; i < input1.size(); i++)
				{
					out.push_back(input1[i] + bias1 + input2[i] + bias2);
				}
			}
		}
		if (activeflag == 1)
		{
			return(sigmoid(out));
		}
		else if (activeflag == 2)
		{
			return(tanh(out));
		}
		else if (activeflag == 3)
		{
			return(Relu(out));
		}
		return out;
	}
	inline vector<Type> upDateValue(vector<Type> input1, vector<Type> input2)
	{
		vector<Type> res;
		if (input1.size() == input2.size())
		{
			for (int i = 0; i < input1.size(); i++)
			{
				res.push_back(input1[i] * input2[i]);
			}
		}

		return res;
	}
	inline vector<Type> upDateValue(vector<Type>& input, Type w[], int activeflag = 1,int outsize=Hidden)//w:15*15
	{
		int num = input.size();
		vector<Type> out;

		if (num!=Datasize&&num!=Hidden)
		{
			return out;
		}
		int j = 0, i = 0;
		for(int i=0;i<outsize;i++)
		{			
			Type res = 0;
			for (j = 0; j < num;j++)
			{
				res += w[i*num +j] * input[j];
			}
			out.push_back(res);
		
		}
		if (activeflag == 1)
		{
			return(sigmoid(out));
		}
		else if (activeflag == 2)
		{
			return(tanh(out));
		}
		else if (activeflag == 3)
		{
			return(Relu(out));
		}
		return out;
	}
	inline vector<Type> upDateValue(Type w[], vector<Type>& input, int activeflag = 1, int outsize =Datasize)
	{
		int num = input.size();
		vector<Type> out;

		if (num != Datasize&&num != Hidden)
		{
			return out;
		}
		int j = 0, i = 0;
		for (int i = 0; i<outsize; i++)
		{
			Type res = 0;
			for (j = 0; j < num; j++)
			{
				res += w[j*num+i] * input[j];
			}
			out.push_back(res);

		}
		if (activeflag == 1)
		{
			return(sigmoid(out));
		}
		else if (activeflag == 2)
		{
			return(tanh(out));
		}
		else if (activeflag == 3)
		{
			return(Relu(out));
		}
		return out;
	}
	inline vector<Type> vectorMulnum(vector<Type>& input, Type num)
	{
		vector<Type> out;
		for (int i = 0; i < input.size(); i++)
		{
			out.push_back(input[i] * num);
		}
		return out;
	}
	inline Type upDateW(Type value, Type loss_delta)
	{
		return value*(1.0 - loss_delta);
	}
	//front
	
	inline bool FrontDoorUPDATE(vector<Type>& door, vector<Type>& doord, vector<Type>& WX, vector<Type>& WH, Type weightx[], Type weighth[])
	{
		if (Datainput.empty() || Hinput.empty() || weightx == NULL || weighth == NULL)
		{
			return false;
		}
		door.clear();
	

		WH.clear();
		WH = upDateValue(Hinput, weighth, -1);
		WX.clear();
		WX = upDateValue(Datainput, weightx, -1);

		door = minusOrplus(WX, WH, 1, 1);

		if (is_training)
		{
			doord.clear();
			doord = sigmoidd(door);
			if (door.size() != Hidden || doord.size() != Hidden)
			{
				return false;
			}
		}	

		if (door.size() != Hidden )
		{
			return false;
		}
		return true;
	}
	inline bool frontHpredict()
	{
		if (Datainput.empty() || outDoorR.empty() || Hinput.empty())
		{
			return false;
		}
		htAndrt.clear();

		htAndrt = upDateValue(Hinput, outDoorR);
		Hpredict.clear();		
		Hpredict = minusOrplus(upDateValue(htAndrt, (Type*)wHPrh, -1), upDateValue(Datainput, (Type*)wHPx, -1), 1, 2);
	  
		if (is_training)
		{
			Hpredictd.clear();
			Hpredictd = tanh(Hpredict);
		}
		
		if (Hpredict.size() != Hidden || Hpredictd.size() != Hidden)
		{
			return false;
		}
		curr_hp_ht = minusOrplus(Hpredict, Hinput, 0);

		return true;
	}
	inline bool frontHoutput()
	{
		if (outDoorZ.empty() || Hinput.empty() || Hpredict.empty())
		{
			return false;
		}
		Houtput.clear();
		Houtput = minusOrplus(upDateValue(outDoorZ, minusOrplus(Hpredict, Hinput, 0)), Hinput, 1);

		return true;
	}
	inline bool frontYoutput()
	{
		if (Hpredict.empty())
		{
			return false;
		}
		Yout.clear();		

		(upDateValue(Hpredict, (Type*)Wout,1,Datasize)).swap(Yout);	

		if (is_training)
		{
			Youtd.clear();
			Youtd = sigmoidd(Yout);
		}	
		return true;
	}
	inline bool get_L2_loss()
	{
		if (Targetinput.size() != Yout.size())
		{
			return false;
		}
		if (is_training)
		{
			Type cur_loss = 0;
			curr_er_ot.clear();
			for (int i = 0; i < Targetinput.size(); i++)
			{
				Type dist = Yout[i] - Targetinput[i];
				dist = error_check(dist);
				curr_er_ot.push_back(dist);
				cur_loss += (dist)*(dist);
			}
			current_loss = cur_loss / 2;
		}

		return true;
	}
	inline Type error_check(Type& dist,Type checkValue=1e-10)
	{
		if (abs(dist) < checkValue)
			if (dist > 0)
				return checkValue;
			else
				return -1.0*(checkValue);
		return dist;

	}
	//back
	inline void upDateNextBack(vector<Type>& fromNext)
	{
		fromNextBack.clear();
		fromNext.swap(fromNextBack);
	}
	inline void Gradient_reset()
	{
		if (step_count != 0)
		{
			return;
		}
		for (int i = 0; i < 6; i++)
		{
			for (int j = 0; j < Hidden; j++)
			{
				Gradient_gather[i][j] = 0;
			}
		}
		for (int k = 0; k < Datasize; k++)
		{
			Gradient_gatherWo[k] = 0;
		}
	}
	inline bool Gradient_StepGather(vector<Type>& newGrad, int idx)
	{
		
		if ((newGrad.size() != Hidden))
		{
			return false;
		}		
		for (int i = 0; i < newGrad.size(); i++)
		{
			Gradient_gather[idx][i] += newGrad[i] * Gradient_weight[step_count];
		}
		return true;
	}
	inline bool Gradient_StepGather(vector<Type>& newGrad)
	{
		if (newGrad.size() != Datasize)
		{
			return false;
		}
		for (int i = 0; i < Datasize; i++)
		{
			Gradient_gatherWo[i] += newGrad[i] * Gradient_weight[step_count];
		}
		return true;
	}
	inline void clear_cur_Grad()
	{
		Grad_Whx.clear();
		Grad_Whrh.clear();
		Grad_Wzx.clear();
		Grad_Wzh.clear();
		Grad_Wrx.clear();
		Grad_Wrh.clear();
		Grad_Wo.clear();
	}
	inline int GatherGradientW()
	{
		clear_cur_Grad();
		vector<Type> tempa,tempb,tempc;		
		
		tempb = vectorMulnum(Youtd, vectorSum(Houtput));
		if (firstOrLast_flag == -1)
		{			
			upDateValue(curr_er_ot, tempb).swap(Grad_Wo);
			tempa = upDateValue(curr_er_ot, Youtd);			
		}	
		else
		{						
			upDateValue(fromNextBack, tempb).swap(Grad_Wo);
			tempa = upDateValue(Youtd, fromNextBack);			
		}			
		if (!(Gradient_StepGather(Grad_Wo)))
			return -4;

		tempa = upDateValue(tempa,(Type*)Wout,-1);
		tempb = upDateValue(upDateValue(tempa, outDoorZ), Hpredictd);
		(vectorMulnum(tempb, vectorSum(Datainput))).swap(Grad_Whx);
		if (firstOrLast_flag != -2)
		{
			(upDateValue((Type*)wHPx, tempb, -1)).swap(tempc);
		}
		(upDateValue(tempb, htAndrt)).swap(Grad_Whrh);	
		if (!(Gradient_StepGather(Grad_Whx, 0) && Gradient_StepGather(Grad_Whrh, 1)))
			return -3;

		(upDateValue(upDateValue(tempa, curr_hp_ht), outDoorZd)).swap(tempb);
		(vectorMulnum(tempb, vectorSum(Datainput))).swap(Grad_Wzx);
		if (firstOrLast_flag != -2)
		{
			tempc = minusOrplus((upDateValue((Type*)wHPx, tempb, -1)), tempc, 1);
		}
		(upDateValue(tempb, Hinput)).swap(Grad_Wzh);
	
		if (!(Gradient_StepGather(Grad_Wzx, 2) && Gradient_StepGather(Grad_Wzh, 3)))
			return -2;

		tempb.clear();
		(upDateValue(upDateValue(Hinput, (Type*)wHPrh, -1), tempa)).swap(tempb);
		tempa.clear();
		(upDateValue(outDoorRd, tempb)).swap(tempa);
		tempb.clear();
		(upDateValue(upDateValue(outDoorZ, tempa),Hpredictd)).swap(tempb);
		(vectorMulnum(tempb, vectorSum(Datainput))).swap(Grad_Wrx);
		if (firstOrLast_flag != -2)
		{
			BackToLastCell.clear();
			(minusOrplus((upDateValue((Type*)wDoorRx, tempb, -1)), tempc, 1)).swap(BackToLastCell);
			if (BackToLastCell.size() != Datasize)
				return -5;			
		}
		(upDateValue(tempb, Hinput)).swap(Grad_Wrh);
		
		if (!(Gradient_StepGather(Grad_Wrx, 4) && Gradient_StepGather(Grad_Wrh, 5)))
			return -1;


		return 0;

	}
	inline void Gradient_clipping()
	{
		Type sumsquare = 0.;
		
		for (int i = 0; i < 6; i++)
		{
			for (int j = 0; j < Hidden; j++)
			{
				sumsquare += (Gradient_gather[i][j] * Gradient_gather[i][j]);
			}
		}
		for (int k = 0; k < Datasize; k++)
		{
			sumsquare += (Gradient_gatherWo[k] * Gradient_gatherWo[k]);
		}

		if (sumsquare >= gradient_tval)
		{
			Type scale= gradient_tval /sumsquare;
			for (int i = 0; i < 6; i++)
			{
				for (int j = 0; j < Hidden; j++)
				{
					Gradient_gather[i][j] *= scale;					
				}
			}
			for (int k = 0; k < Datasize; k++)
			{
				Gradient_gatherWo[k] *= scale;
			}
		}
	}
	inline bool UpDateWeights()
	{
		if (step_count != learning_step)
		{
			return false;
		}
		Gradient_clipping();		
		for (int i = 0; i < Hidden; i++)
		{
			if(Hidden>=Datasize)
			{
				for (int j = 0; j < Datasize; j++)
				{
					wHPrh[i][j] = upDateW(wHPrh[i][j], Gradient_gather[0][j]);
					wDoorZh[i][j] = upDateW(wDoorZh[i][j], Gradient_gather[3][j]);
					wDoorRh[i][j] = upDateW(wDoorRh[i][j], Gradient_gather[5][j]);
					wHPx[i][j] = upDateW(wHPx[i][j], Gradient_gather[1][j]);
					wDoorRx[i][j] = upDateW(wDoorRx[i][j], Gradient_gather[4][j]);
					wDoorZx[i][j] = upDateW(wDoorZx[i][j], Gradient_gather[2][j]);
					Wout[j][i] = upDateW(Wout[j][i], Gradient_gatherWo[j]);
				}
				for(int k=Datasize;k<Hidden;k++)
				{
					wHPrh[i][k] = upDateW(wHPrh[i][k], Gradient_gather[0][k]);
					wDoorZh[i][k] = upDateW(wDoorZh[i][k], Gradient_gather[3][k]);
					wDoorRh[i][k] = upDateW(wDoorRh[i][k], Gradient_gather[5][k]);			
					
				}				
			}
			else
			{
				for (int j = 0; j < Hidden; j++)
				{
					wHPrh[i][j] = upDateW(wHPrh[i][j], Gradient_gather[0][j]);
					wDoorZh[i][j] = upDateW(wDoorZh[i][j], Gradient_gather[3][j]);
					wDoorRh[i][j] = upDateW(wDoorRh[i][j], Gradient_gather[5][j]);
					wHPx[i][j] = upDateW(wHPx[i][j], Gradient_gather[1][j]);
					wDoorRx[i][j] = upDateW(wDoorRx[i][j], Gradient_gather[4][j]);
					wDoorZx[i][j] = upDateW(wDoorZx[i][j], Gradient_gather[2][j]);
					Wout[j][i] = upDateW(Wout[j][i], Gradient_gatherWo[j]);
				}
				for (int k = Hidden; k<Datasize; k++)
				{
					wHPx[i][k] = upDateW(wHPx[i][k], Gradient_gather[1][k]);
					wDoorRx[i][k] = upDateW(wDoorRx[i][k], Gradient_gather[4][k]);
					wDoorZx[i][k] = upDateW(wDoorZx[i][k], Gradient_gather[2][k]);
					Wout[k][i] = upDateW(Wout[k][i], Gradient_gatherWo[k]);

				}
			}			
		}
		return true;
	}


};

/****************************************MultiGRU Class************************************************/
class MultiGRU
{
public:
	MultiGRU(int predict_length, int pre_start_day, int Layer_num=Layernum, Type error_permit = Error_limit, Type rate = Learning_rate, int learning_st = Learning_step, Type clip_gradient = Gradient_tval, int init = 1)
	{
		layer_num = Layer_num;
		if (layer_num <= 1)
		{
			MultiSet = false;
			return;
		}
		

		Global_count = 0;
		current_loss = 0;
		loss_permit = error_permit;
		learning_step = learning_st;
		L_rate = rate;
		predict_l = predict_length;
		date_interval = pre_start_day;
		is_training = false;
		gradient_tval = clip_gradient;


		int flag;
		for (int i = 0; i < Layer_num; i++)
		{
			flag = i;
			if (i == Layer_num - 1)
			{
				flag = -1;
			}
			else if (i == 0)
			{
				flag = -2;
			}
			GRUcell temp(flag, rate, learning_st, clip_gradient / layer_num, init);
			Layers.push_back(temp);
		}

	}
	void train(vector<vector<InType >>& input_data, int epoch = max_epoch)
	{
		
		is_training = true;
		int data_length = input_data[0].size() - 1;
		if (data_length <= 0)
		{
			printf("Lack of training data ,MAN.");
			return;
		}
		if (!cookData(input_data))
		{
			printf("Data bad cooked ,MAN.");
			return;
		}

		while (is_training &&Global_count <max_epoch)
		{
			vector<Type> Xinput, Ytarget;
			for (int j = 0; j < data_length - 1&&is_training; j++)
			{
				Xinput.clear();
				Ytarget.clear();
				for (int i = 0; i < Datasize; i++)
				{
					Xinput.push_back(wholeData[i][j]);
					Ytarget.push_back(wholeData[i][j + 1]);
				}
				
				if (one_turn(Xinput, Ytarget) != 0)
				{
					is_training = false;
					break;
				}
					
			}
			Global_count++;
			//printf("__%d", Global_count);
		}
		is_training = false;
		predict();
		
	}
	void predict()
	{
		if (is_training || last_sevenPack.size() != Datasize)
		{
			return;
		}

		final_result.clear();
		if (FrontPro(last_sevenPack) == 0)
		{
			int j;
			for (j = 0; j < predict_l - 1 + date_interval; j++)
			{			
				FrontPro(Layers[layer_num - 1].Yout);
			}
			if (j == predict_l - 1 + date_interval)
				for (int i = 0; i < Layers[layer_num - 1].Yout.size(); i++)
				{
					Type temp = Layers[layer_num - 1].Yout[i] * decodeValue + 1.5;
					
					printf("!%d", temp);
					final_result.push_back(InType(temp)-1);
					
					
				}
		}
		else
			printf("Predict Failed");
	}

	/*************************************************************************************/
	vector<vector<Type>> wholeData;
	vector<Type> last_sevenPack;
	vector<InType > final_result;

	vector<Type> Datainput;
	vector<Type> Targetinput;
	int Global_count;
	
	
protected:

	bool cookData(vector<vector<InType >>& input_data)
	{
		DataServer dataserver;
		wholeData.clear();
		last_sevenPack.clear();

		(dataserver.get_Cooked_value(input_data)).swap(wholeData);
		(dataserver.get_LastWeek()).swap(last_sevenPack);

		decodeValue = dataserver.TopValue;
		if (decodeValue > 0. && !wholeData.empty())
			return true;
		return false;
	}
	void fillin_data(vector<Type>& input1, vector<Type>& input2)
	{
		Datainput.clear();
		input1.swap(Datainput);
		Targetinput.clear();
		input2.swap(Targetinput);
	}
	int one_turn(vector<Type>& input1, vector<Type>& input2)
	{
		fillin_data(input1, input2);
		
		if (FrontPro(Datainput) != 0)
			return -3;
		if (BackPro() != 0)
			return -2;
		step_count++;
		current_loss = Layers[layer_num - 1].current_loss;
	
		if (current_loss <= loss_permit)
		{
			is_training = false;
			return -1;
		}
		return 0;
	}
	int FrontPro(vector<Type>& input )
	{		
		Layers[0].fillin_data(input);
		if (Layers[0].FrontPro(is_training) == 0)
		{
			for (int i = 1; i < layer_num - 1; i++)
			{
				Layers[i].fillin_data(Layers[i - 1].Yout);
				if (Layers[i].FrontPro(is_training) != 0)
				{
					return -1;
				}
			}
		}		
		Layers[layer_num - 1].fillin_data(Layers[layer_num - 2].Yout, Targetinput);
		return(Layers[layer_num - 1].FrontPro(is_training));
	}
	int BackPro()
	{
		if (!is_training)
			return 0;
		for (int i = layer_num - 1; i >0; i--)
		{
			if (Layers[i].BackPro() != 0)
				return -1;			
			(Layers[i - 1].fromNextBack).clear();
			(Layers[i].BackToLastCell).swap(Layers[i - 1].fromNextBack);			
		}
		return(Layers[0].BackPro());
	}

private:
	int layer_num;
	bool is_training;
	int learning_step;
	
	int step_count;
	int predict_l;
	int date_interval;

	Type decodeValue;
	Type gradient_tval;
	Type current_loss;
	Type loss_permit;
	Type L_rate;
	vector<GRUcell> Layers;
};