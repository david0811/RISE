//T-Files
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TLatex.h"
#include "TLine.h"
#include "TMath.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TBox.h"
#include "TAxis.h"

//RooFit-Files
#include "RooDataSet.h"
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooAbsPdf.h"
#include "RooPlot.h"

//Sonstiges
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

// Boost
#include <boost/lexical_cast.hpp>

// from LHCbphi_s
#include <doofit/builder/EasyPdf/EasyPdf.h> // doosoft/doofit/ROOT_v6.02.08/include/doofit/builder/EasyPdf/
#include <doofit/fitter/easyfit/EasyFit.h> // doosoft/doofit/ROOT_v6.02.08/include/doofit/fitter/easyfit/
#include <doocore/io/EasyTuple.h> // doosoft/doocore/ROOT_v6.02.08/include/doocore/io/
#include <doocore/lutils/lutils.h> // doosoft/doocore/ROOT_v6.02.08/include/doocore/lutils/
#include <doofit/fitter/splot/SPlotFit2.h> // doosoft/doofit/ROOT_v6.02.08/include/doofit/fitter/splot/
#include <dooselection/reducer/SPlotterReducer.h> // doosoft/dooselection/ROOT_v6.02.08/include/dooselection/reducer/

using namespace RooFit;
using namespace doocore;
using namespace doofit;

void addLineToRooPlot(double massPoint, RooPlot &plotObject, int lineColorNumber = 2, int lineStyleNumber = 2){
    TLine *temp_line = new TLine(massPoint, plotObject.GetMinimum(), massPoint, plotObject.GetMaximum());
    temp_line->SetLineColor(lineColorNumber);
    temp_line->SetLineStyle(lineStyleNumber);
    plotObject.addObject(temp_line);
    
    //delete temp_line;
}

void addLineToRooPlot(double mean, double sigma, RooPlot &plotObject, int lineColorNumber = 2, int lineStyleNumber = 2, int sigmaCut = 3){
    TLine *temp_line_low = new TLine(mean-sigma*sigmaCut, plotObject.GetMinimum(), mean-sigma*sigmaCut, plotObject.GetMaximum());
    TLine *temp_line_high = new TLine(mean+sigma*sigmaCut, plotObject.GetMinimum(), mean+sigma*sigmaCut, plotObject.GetMaximum());
    
    temp_line_low->SetLineColor(lineColorNumber);
    temp_line_low->SetLineStyle(lineStyleNumber);
    temp_line_high->SetLineColor(lineColorNumber);
    temp_line_high->SetLineStyle(lineStyleNumber);
    
    plotObject.addObject(temp_line_low);
    plotObject.addObject(temp_line_high);
    
    //delete temp_line_low;
    //delete temp_line_high;
}

double fomMaximizer(TTree* dataTree, TTree* mcTree, RooRealVar* bs_mass_data, RooRealVar* bdt_classifier_data, std::string savePlotPath, std::vector<double> cutpoint,
    std::vector<double> &signalCandidates, std::vector<double> &signalyield, std::vector<double> &backgroundCandidates, double bkgmin, double bkgmax, double signalMean, double signalSigma){
    for(auto value: cutpoint){
        doocore::io::EasyTuple dataTuple(dataTree, RooArgSet(*bs_mass_data, *bdt_classifier_data));
        TString cutString =  "BDTG_classifier > " + std::to_string(value);
        RooDataSet& data = dataTuple.ConvertToDataSet(RooFit::Cut(cutString));
        doofit::builder::EasyPdf myPdf;

        //bs_mass_data->setRange("lowerRange", bkgmin, signalMean-signalSigma);
        //bs_mass_data->setRange("upperRange", signalMean+signalSigma, bkgmax);
        bs_mass_data->setRange("backgroundRange", bkgmin, bkgmax);
        bs_mass_data->setRange("signalRange", signalMean-signalSigma, signalMean+signalSigma);

        myPdf.Exponential("expo1", *bs_mass_data, myPdf.Var("lambda_1"));
        myPdf.Gaussian("gauss1",*bs_mass_data, myPdf.Var("sigmean"), myPdf.Var("sigwidth"));
        //myPdf.Gaussian("gauss2",*bs_mass_data, myPdf.Var("sigmean1"), myPdf.Var("sigwidth1"));

        myPdf.Add("fitModell", RooArgList(myPdf.Pdf("expo1"), myPdf.Pdf("gauss1")), RooArgList(myPdf.Var("expo1_yield"), myPdf.Var("gauss1_yield")));

        myPdf.Pdf("fitModell").getParameters(&data)->readFromFile("/home/dlafferty/bachelor-template/david/results/FoM/Init_MC_Bs2phiphi.txt");
        myPdf.Pdf("fitModell").fitTo(data, Save(true), NumCPU(6), Extended(true), Range("backgroundRange"));
        myPdf.Pdf("fitModell").getParameters(&data)->writeToFile("/home/dlafferty/bachelor-template/david/results/FoM/Output_MC_Bs2phiphi.txt");

        RooPlot* bs_massplot_data = bs_mass_data->frame(Bins(100));
        data.plotOn(bs_massplot_data);
        myPdf.Pdf("fitModell").plotOn(bs_massplot_data, Components(RooArgSet(myPdf.Pdf("expo1"))),LineStyle(3),LineColor(3));
        myPdf.Pdf("fitModell").plotOn(bs_massplot_data, Components(RooArgSet(myPdf.Pdf("gauss1"))),LineStyle(4),LineColor(4));
        //myPdf.Pdf("fitModell").plotOn(bs_massplot_data, Components(RooArgSet(myPdf.Pdf("gauss2"))),LineStyle(5),LineColor(5));
        myPdf.Pdf("fitModell").plotOn(bs_massplot_data);

        addLineToRooPlot(signalMean-signalSigma, *bs_massplot_data);
        addLineToRooPlot(signalMean+signalSigma, *bs_massplot_data);
        
        TBox myBox(signalMean-signalSigma, bs_massplot_data->GetMinimum(), signalMean+signalSigma, bs_massplot_data->GetMaximum());
        myBox.SetFillStyle(3003);
        myBox.SetFillColor(2);
        bs_massplot_data->addObject(&myBox);

        TString saveString = "bs_mass_sidebands_after_classifier_cut_" + std::to_string(value);
        if(saveString.Contains("-")){
            saveString.ReplaceAll("-","minus_");
        }
        TLatex* fitLabel = new TLatex(0.25, 0.8, "#splitline{#scale[0.7]{LHCb unofficial}}{#scale[0.7]{Data, #sqrt{s} = 7 + 8 TeV}}"); 

        doocore::lutils::PlotSimple(saveString.ReplaceAll(".", "_"), bs_massplot_data, *fitLabel, savePlotPath);
       
        //if the pdf is normalized
        double background_yield = (myPdf.Pdf("fitModell").createIntegral(*bs_mass_data, *bs_mass_data, "signalRange"))->getVal() * 
        (myPdf.Var("expo1_yield").getVal());
        backgroundCandidates.push_back(background_yield);
        double signal_yield = (myPdf.Pdf("fitModell").createIntegral(*bs_mass_data, *bs_mass_data, "signalRange"))->getVal() * 
        (myPdf.Var("gauss1_yield").getVal());  //+ myPdf.Var("gauss2_yield").getVal());
        signalyield.push_back(signal_yield);
        cutString = cutString + "&&" + "abs(B_s0_MM-" + std::to_string(signalMean) + ")<" + std::to_string(signalSigma);
        signalCandidates.push_back(mcTree->GetEntries(cutString));
    }
}
 
int main(int argc, char* argv[]){
    doocore::lutils::setStyle();

    TFile dataFile("/net/storage03/data/users/dlafferty/NTuples/data/2012/combined/Bs2phiphi_data_2012_corrected_TupleA__BDT_241114_1.root", "READ");
    TFile mcFile("/net/storage03/data/users/dlafferty/NTuples/SignalMC/2012/combined/Bs2phiphi_MC_2012_combined_corrected_TupleA__BDT_241114_1.root", "READ");
    
    TString inputTree = "DecayTree";
    std::string savePlotPath = "/home/dlafferty/bachelor-template/david/results/FoM/";
    
    TTree* dataTree = (TTree*)dataFile.Get(inputTree);
    TTree* mcTree = (TTree*)mcFile.Get(inputTree);
    
    double bkgmin = 5080;
    double bkgmax = 5480;
    double signalMean = 5368;
    double signalSigma = 50;

    RooRealVar * bs_mass_data = new RooRealVar("B_s0_MM", "m_{#it{K}#it{K}#it{K}#it{K}}", bkgmin, bkgmax, "MeV/c^{2}"); 
    RooRealVar * bdt_classifier_data = new RooRealVar("BDTG_classifier", "BDTG_classifier", -1., 1.); 

    std::string cutstring = "abs(B_s0_MM-" + std::to_string(signalMean) + ")<" + std::to_string(signalSigma);
    double mcEntries = mcTree->GetEntries(cutstring.c_str());

    // --- Compute Punzi FoM ---
    std::vector<double> bdtCutPoints, signalCandidates, signalyield, backgroundCandidates;

    for(int i = 0; i<=100; i++) {
            bdtCutPoints.push_back(-1. + i/50.);
    }

    fomMaximizer(dataTree, mcTree, bs_mass_data, bdt_classifier_data, savePlotPath, bdtCutPoints, signalCandidates, signalyield, backgroundCandidates, bkgmin, bkgmax, signalMean, signalSigma);
    
    const int sig = signalCandidates.size();  
    const int bck = backgroundCandidates.size();

    Double_t x[sig], y[bck];

    double punziNumber = 5.0;
    double punziMax = (signalCandidates[0]/mcEntries)/((punziNumber/2.0)+(TMath::Sqrt(backgroundCandidates[0])));
    //double punziMax = (signalCandidates[0]/mcEntries)/(TMath::Sqrt(signalCandidates[0]+backgroundCandidates[0]));
    double best_cut = bdtCutPoints[0];

    ofstream myfile;
    myfile.open ("/home/dlafferty/bachelor-template/david/results/FoM/BDTG_reweight_results_zoom.txt");

    for(int i = 0; i < signalCandidates.size(); i++){
        //double punziFoM = (signalCandidates[i]/mcEntries)/(TMath::Sqrt(signalCandidates[i]+backgroundCandidates[i]));
        double punziFoM = (signalCandidates[i]/mcEntries)/((punziNumber/2.0)+(TMath::Sqrt(backgroundCandidates[i])));
        myfile << "Cut: " + std::to_string(bdtCutPoints[i]) + " Sig cands: " + std::to_string(signalCandidates[i]) + " Bkg cands: " + std::to_string(backgroundCandidates[i]) + " Punzi-Result: "
        + std::to_string(punziFoM) + " Signal yield: " + std::to_string(signalyield[i]) << std::endl;
        x[i] = bdtCutPoints[i];
        y[i] = punziFoM;
        if (punziFoM > punziMax) {
                punziMax = punziFoM;
                best_cut = bdtCutPoints[i];
        }
    }

    // --- Repeat for interesting region ---
    std::vector<double> bdtCutPoints1, signalCandidates1, signalyield1, backgroundCandidates1;
    
    for(int i = 0; i<=40; i++) {
        bdtCutPoints1.push_back(best_cut-0.02 + i/1000.);
    }

    fomMaximizer(dataTree, mcTree, bs_mass_data, bdt_classifier_data, savePlotPath, bdtCutPoints1, signalCandidates1, signalyield1, backgroundCandidates1, bkgmin, bkgmax, signalMean, signalSigma);

    const int sig1 = signalCandidates1.size();
    const int bck1 = backgroundCandidates1.size();

    Double_t x1[sig1], y1[bck1];

    double punziMax1 = (signalCandidates1[0]/mcEntries)/((punziNumber/2.0)+(TMath::Sqrt(backgroundCandidates1[0])));
    //double punziMax1 = (signalCandidates1[0]/mcEntries)/(TMath::Sqrt(signalCandidates1[0]+backgroundCandidates1[0]));
    double best_cut1 = bdtCutPoints1[0];

    // --- Display results ---
    myfile << "\n" << std::endl;
    
    for(int i = 0; i < signalCandidates1.size(); i++){
        //double punziFoM1 = (signalCandidates1[i]/mcEntries)/(TMath::Sqrt(signalCandidates1[i]+backgroundCandidates1[i]));
        double punziFoM1 = (signalCandidates1[i]/mcEntries)/((punziNumber/2.0)+(TMath::Sqrt(backgroundCandidates1[i])));
        myfile << "Cut: " + std::to_string(bdtCutPoints1[i]) + " Sig cands: " + std::to_string(signalCandidates1[i]) + " Bkg cands: " + std::to_string(backgroundCandidates1[i]) + " Punzi-Result: "
            + std::to_string(punziFoM1) + " Signal yield: " + std::to_string(signalyield1[i]) << std::endl;
        x1[i] = bdtCutPoints1[i];
        y1[i] = punziFoM1;
        if (punziFoM1 > punziMax1) {
                punziMax1 = punziFoM1;
                best_cut1 = bdtCutPoints1[i];
        }
    }
    
    TCanvas *c1 = new TCanvas("c1", "Maximising the Figure of Merit", 800, 600);
    c1->cd();
    TGraph *gr = new TGraph(sig, x, y);
    gr->GetXaxis()->SetTitle("Cut point");
    gr->GetYaxis()->SetTitle("Punzi Result");
    gr->Draw("AC*");
    c1->SaveAs("/home/dlafferty/bachelor-template/david/results/FoM/Punzi_BDTG_reweight.pdf");

    
    TCanvas *c2 = new TCanvas("c2", "Maximising the Figure of Merit", 800, 600);
    c2->cd();
    TGraph *gr1 = new TGraph(sig1, x1, y1);
    gr1->GetXaxis()->SetTitle("Cut point");
    gr1->GetYaxis()->SetTitle("Punzi Result");
    gr1->Draw("AC*");
    c2->SaveAs("/home/dlafferty/bachelor-template/david/results/FoM/Punzi_BDTG_reweight_zoom.pdf");
    

    myfile << "\n" << std::endl;
    myfile << "Optimal cut point: " << best_cut << std::endl;
    myfile << "Optimal cut point (with zoom): " << best_cut1 << std::endl;

    myfile.close();

    return 0;
}
