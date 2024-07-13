
/*********************                                                        */
/*! \file Marabou.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** \brief [[ Add one-line brief description here ]]
 **
 ** [[ Add lengthier description here ]]
 **/

#include "AcasParser.h"
#include "AutoFile.h"
#include "GlobalConfiguration.h"
#include "Tableau.h"
#include "File.h"
#include "MStringf.h"
#include "Marabou.h"
#include "Options.h"
#include "PropertyParser.h"
#include "MarabouError.h"
#include "QueryLoader.h"
#include "Equation.h"
#include "MinConstraint.h"
#include "float.h"
#ifdef _WIN32
#undef ERROR
#endif
double log(unsigned n);

Marabou::Marabou()
        : _acasParser(NULL), _engine()
{
}

Marabou::~Marabou()
{
    if (_acasParser)
    {
        delete _acasParser;
        _acasParser = NULL;
    }
}

void Marabou::run()
{
    struct timespec start = TimeUtils::sampleMicro();

    prepareInputQuery();
    solveQuery();

    struct timespec end = TimeUtils::sampleMicro();

    unsigned long long totalElapsed = TimeUtils::timePassed(start, end);
    displayResults(totalElapsed);

    if (Options::get()->getBool(Options::EXPORT_ASSIGNMENT))
        exportAssignment();
}


unsigned Marabou::sigmoid_anagha_final(unsigned variable)
{
    //we want sigmoid variable - lse_approximation
    Set<unsigned int> set3, set4, minSet3, minSet4;

    unsigned q2_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q2_ + 1);
    _inputQuery.setLowerBound(q2_, -100.0);
    _inputQuery.setUpperBound(q2_, 100.0);
    Equation eq2_;
    eq2_.addAddend(1, q2_);
    eq2_.addAddend(-0.002543865904564067, variable);
    eq2_.setScalar(0.01805060314480649);
    _inputQuery.addEquation(eq2_);
    set3.insert(q2_);
    unsigned q3_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q3_ + 1);
    _inputQuery.setLowerBound(q3_, -100.0);
    _inputQuery.setUpperBound(q3_, 100.0);
    Equation eq3_;
    eq3_.addAddend(1, q3_);
    eq3_.addAddend(-0.008671798219180675, variable);
    eq3_.setScalar(0.05003576561700055);
    _inputQuery.addEquation(eq3_);
    set3.insert(q3_);
    unsigned q4_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q4_ + 1);
    _inputQuery.setLowerBound(q4_, -100.0);
    _inputQuery.setUpperBound(q4_, 100.0);
    Equation eq4_;
    eq4_.addAddend(1, q4_);
    eq4_.addAddend(-0.0160918859304203, variable);
    eq4_.setScalar(0.08240230673319072);
    _inputQuery.addEquation(eq4_);
    set3.insert(q4_);
    unsigned q5_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q5_ + 1);
    _inputQuery.setLowerBound(q5_, -100.0);
    _inputQuery.setUpperBound(q5_, 100.0);
    Equation eq5_;
    eq5_.addAddend(1, q5_);
    eq5_.addAddend(-0.02434297406286057, variable);
    eq5_.setScalar(0.1143649740425666);
    _inputQuery.addEquation(eq5_);
    set3.insert(q5_);
    unsigned q6_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q6_ + 1);
    _inputQuery.setLowerBound(q6_, -100.0);
    _inputQuery.setUpperBound(q6_, 100.0);
    Equation eq6_;
    eq6_.addAddend(1, q6_);
    eq6_.addAddend(-0.03648513005820122, variable);
    eq6_.setScalar(0.15616581066719448);
    _inputQuery.addEquation(eq6_);
    set3.insert(q6_);
    unsigned q7_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q7_ + 1);
    _inputQuery.setLowerBound(q7_, -100.0);
    _inputQuery.setUpperBound(q7_, 100.0);
    Equation eq7_;
    eq7_.addAddend(1, q7_);
    eq7_.addAddend(-0.053930892473068974, variable);
    eq7_.setScalar(0.20870845479245959);
    _inputQuery.addEquation(eq7_);
    set3.insert(q7_);
    unsigned q8_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q8_ + 1);
    _inputQuery.setLowerBound(q8_, -100.0);
    _inputQuery.setUpperBound(q8_, 100.0);
    Equation eq8_;
    eq8_.addAddend(1, q8_);
    eq8_.addAddend(-0.0712222309563578, variable);
    eq8_.setScalar(0.25402251637804774);
    _inputQuery.addEquation(eq8_);
    set3.insert(q8_);
    unsigned q9_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q9_ + 1);
    _inputQuery.setLowerBound(q9_, -100.0);
    _inputQuery.setUpperBound(q9_, 100.0);
    Equation eq9_;
    eq9_.addAddend(1, q9_);
    eq9_.addAddend(-0.08517649902476777, variable);
    eq9_.setScalar(0.28712509843219053);
    _inputQuery.addEquation(eq9_);
    set3.insert(q9_);
    unsigned q10_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q10_ + 1);
    _inputQuery.setLowerBound(q10_, -100.0);
    _inputQuery.setUpperBound(q10_, 100.0);
    Equation eq10_;
    eq10_.addAddend(1, q10_);
    eq10_.addAddend(-0.10105713280442323, variable);
    eq10_.setScalar(0.32137391451541486);
    _inputQuery.addEquation(eq10_);
    set3.insert(q10_);
    unsigned q11_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q11_ + 1);
    _inputQuery.setLowerBound(q11_, -100.0);
    _inputQuery.setUpperBound(q11_, 100.0);
    Equation eq11_;
    eq11_.addAddend(1, q11_);
    eq11_.addAddend(-0.11877226533845178, variable);
    eq11_.setScalar(0.35576112149866723);
    _inputQuery.addEquation(eq11_);
    set3.insert(q11_);
    unsigned q12_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q12_ + 1);
    _inputQuery.setLowerBound(q12_, -100.0);
    _inputQuery.setUpperBound(q12_, 100.0);
    Equation eq12_;
    eq12_.addAddend(1, q12_);
    eq12_.addAddend(-0.13805388170120814, variable);
    eq12_.setScalar(0.38903490752059966);
    _inputQuery.addEquation(eq12_);
    set3.insert(q12_);
    unsigned q13_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q13_ + 1);
    _inputQuery.setLowerBound(q13_, -100.0);
    _inputQuery.setUpperBound(q13_, 100.0);
    Equation eq13_;
    eq13_.addAddend(1, q13_);
    eq13_.addAddend(-0.15841276484258954, variable);
    eq13_.setScalar(0.4197833653601326);
    _inputQuery.addEquation(eq13_);
    set3.insert(q13_);
    unsigned q14_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q14_ + 1);
    _inputQuery.setLowerBound(q14_, -100.0);
    _inputQuery.setUpperBound(q14_, 100.0);
    Equation eq14_;
    eq14_.addAddend(1, q14_);
    eq14_.addAddend(-0.17911003633714492, variable);
    eq14_.setScalar(0.44658826147689556);
    _inputQuery.addEquation(eq14_);
    set3.insert(q14_);
    unsigned q15_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q15_ + 1);
    _inputQuery.setLowerBound(q15_, -100.0);
    _inputQuery.setUpperBound(q15_, 100.0);
    Equation eq15_;
    eq15_.addAddend(1, q15_);
    eq15_.addAddend(-0.1991616624642956, variable);
    eq15_.setScalar(0.46824484877026984);
    _inputQuery.addEquation(eq15_);
    set3.insert(q15_);
    unsigned q16_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q16_ + 1);
    _inputQuery.setLowerBound(q16_, -100.0);
    _inputQuery.setUpperBound(q16_, 100.0);
    Equation eq16_;
    eq16_.addAddend(1, q16_);
    eq16_.addAddend(-0.217391566491753, variable);
    eq16_.setScalar(0.48401859038912826);
    _inputQuery.addEquation(eq16_);
    set3.insert(q16_);
    unsigned q17_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q17_ + 1);
    _inputQuery.setLowerBound(q17_, -100.0);
    _inputQuery.setUpperBound(q17_, 100.0);
    Equation eq17_;
    eq17_.addAddend(1, q17_);
    eq17_.addAddend(-0.23254124615360464, variable);
    eq17_.setScalar(0.49388113246220744);
    _inputQuery.addEquation(eq17_);
    set3.insert(q17_);
    unsigned q18_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q18_ + 1);
    _inputQuery.setLowerBound(q18_, -100.0);
    _inputQuery.setUpperBound(q18_, 100.0);
    Equation eq18_;
    eq18_.addAddend(1, q18_);
    eq18_.addAddend(-0.2465459176469087, variable);
    eq18_.setScalar(0.4996724297871519);
    _inputQuery.addEquation(eq18_);
    set3.insert(q18_);


    unsigned q101_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q101_ + 1);
    _inputQuery.setLowerBound(q101_, -100.0);
    _inputQuery.setUpperBound(q101_, 100.0);
    Equation eq101_;
    eq101_.addAddend(-1, q101_);
    eq101_.addAddend(-0.2465459176469087, variable);
    eq101_.setScalar(0.5003275702128485);
    _inputQuery.addEquation(eq101_);
    set4.insert(q101_);
    unsigned q102_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q102_ + 1);
    _inputQuery.setLowerBound(q102_, -100.0);
    _inputQuery.setUpperBound(q102_, 100.0);
    Equation eq102_;
    eq102_.addAddend(-1, q102_);
    eq102_.addAddend(-0.2251459482999449, variable);
    eq102_.setScalar(0.5101081756873811);
    _inputQuery.addEquation(eq102_);
    set4.insert(q102_);
    unsigned q103_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q103_ + 1);
    _inputQuery.setLowerBound(q103_, -100.0);
    _inputQuery.setUpperBound(q103_, 100.0);
    Equation eq103_;
    eq103_.addAddend(-1, q103_);
    eq103_.addAddend(-0.1991616624642951, variable);
    eq103_.setScalar(0.5317551512297305);
    _inputQuery.addEquation(eq103_);
    set4.insert(q103_);
    unsigned q104_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q104_ + 1);
    _inputQuery.setLowerBound(q104_, -100.0);
    _inputQuery.setUpperBound(q104_, 100.0);
    Equation eq104_;
    eq104_.addAddend(-1, q104_);
    eq104_.addAddend(-0.17911003633714528, variable);
    eq104_.setScalar(0.5534117385231038);
    _inputQuery.addEquation(eq104_);
    set4.insert(q104_);
    unsigned q105_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q105_ + 1);
    _inputQuery.setLowerBound(q105_, -100.0);
    _inputQuery.setUpperBound(q105_, 100.0);
    Equation eq105_;
    eq105_.addAddend(-1, q105_);
    eq105_.addAddend(-0.15841276484258868, variable);
    eq105_.setScalar(0.5802166346398685);
    _inputQuery.addEquation(eq105_);
    set4.insert(q105_);
    unsigned q106_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q106_ + 1);
    _inputQuery.setLowerBound(q106_, -100.0);
    _inputQuery.setUpperBound(q106_, 100.0);
    Equation eq106_;
    eq106_.addAddend(-1, q106_);
    eq106_.addAddend(-0.13805388170120766, variable);
    eq106_.setScalar(0.6109650924794012);
    _inputQuery.addEquation(eq106_);
    set4.insert(q106_);
    unsigned q107_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q107_ + 1);
    _inputQuery.setLowerBound(q107_, -100.0);
    _inputQuery.setUpperBound(q107_, 100.0);
    Equation eq107_;
    eq107_.addAddend(-1, q107_);
    eq107_.addAddend(-0.11877226533845185, variable);
    eq107_.setScalar(0.6442388785013329);
    _inputQuery.addEquation(eq107_);
    set4.insert(q107_);
    unsigned q108_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q108_ + 1);
    _inputQuery.setLowerBound(q108_, -100.0);
    _inputQuery.setUpperBound(q108_, 100.0);
    Equation eq108_;
    eq108_.addAddend(-1, q108_);
    eq108_.addAddend(-0.10983042222425944, variable);
    eq108_.setScalar(0.6606306476130153);
    _inputQuery.addEquation(eq108_);
    set4.insert(q108_);
    unsigned q109_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q109_ + 1);
    _inputQuery.setLowerBound(q109_, -100.0);
    _inputQuery.setUpperBound(q109_, 100.0);
    Equation eq109_;
    eq109_.addAddend(-1, q109_);
    eq109_.addAddend(-0.040470623980445174, variable);
    eq109_.setScalar(0.8313480269902461);
    _inputQuery.addEquation(eq109_);
    set4.insert(q109_);
    unsigned q110_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q110_ + 1);
    _inputQuery.setLowerBound(q110_, -100.0);
    _inputQuery.setUpperBound(q110_, 100.0);
    Equation eq110_;
    eq110_.addAddend(-1, q110_);
    eq110_.addAddend(-0.03229299282395175, variable);
    eq110_.setScalar(0.8578503415691678);
    _inputQuery.addEquation(eq110_);
    set4.insert(q110_);
    unsigned q111_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q111_ + 1);
    _inputQuery.setLowerBound(q111_, -100.0);
    _inputQuery.setUpperBound(q111_, 100.0);
    Equation eq111_;
    eq111_.addAddend(-1, q111_);
    eq111_.addAddend(-0.025668092163887983, variable);
    eq111_.setScalar(0.8809372759083436);
    _inputQuery.addEquation(eq111_);
    set4.insert(q111_);
    unsigned q112_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q112_ + 1);
    _inputQuery.setLowerBound(q112_, -100.0);
    _inputQuery.setUpperBound(q112_, 100.0);
    Equation eq112_;
    eq112_.addAddend(-1, q112_);
    eq112_.addAddend(-0.02033951725545943, variable);
    eq112_.setScalar(0.9008072241656719);
    _inputQuery.addEquation(eq112_);
    set4.insert(q112_);
    unsigned q113_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q113_ + 1);
    _inputQuery.setLowerBound(q113_, -100.0);
    _inputQuery.setUpperBound(q113_, 100.0);
    Equation eq113_;
    eq113_.addAddend(-1, q113_);
    eq113_.addAddend(-0.016077813562956105, variable);
    eq113_.setScalar(0.9177391169334725);
    _inputQuery.addEquation(eq113_);
    set4.insert(q113_);
    unsigned q114_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q114_ + 1);
    _inputQuery.setLowerBound(q114_, -100.0);
    _inputQuery.setUpperBound(q114_, 100.0);
    Equation eq114_;
    eq114_.addAddend(-1, q114_);
    eq114_.addAddend(-0.012684539525458829, variable);
    eq114_.setScalar(0.932049037105814);
    _inputQuery.addEquation(eq114_);
    set4.insert(q114_);
    unsigned q115_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q115_ + 1);
    _inputQuery.setLowerBound(q115_, -100.0);
    _inputQuery.setUpperBound(q115_, 100.0);
    Equation eq115_;
    eq115_.addAddend(-1, q115_);
    eq115_.addAddend(-0.00890273748673438, variable);
    eq115_.setScalar(0.9490553646042065);
    _inputQuery.addEquation(eq115_);
    set4.insert(q115_);
    unsigned q116_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q116_ + 1);
    _inputQuery.setLowerBound(q116_, -100.0);
    _inputQuery.setUpperBound(q116_, 100.0);
    Equation eq116_;
    eq116_.addAddend(-1, q116_);
    eq116_.addAddend(-0.005501670887814405, variable);
    eq116_.setScalar(0.9658501156231741);
    _inputQuery.addEquation(eq116_);
    set4.insert(q116_);
    unsigned q117_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q117_ + 1);
    _inputQuery.setLowerBound(q117_, -100.0);
    _inputQuery.setUpperBound(q117_, 100.0);
    Equation eq117_;
    eq117_.addAddend(-1, q117_);
    eq117_.addAddend(-0.003390707677573013, variable);
    eq117_.setScalar(0.9773047373082717);
    _inputQuery.addEquation(eq117_);
    set4.insert(q117_);
    unsigned q118_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q118_ + 1);
    _inputQuery.setLowerBound(q118_, -100.0);
    _inputQuery.setUpperBound(q118_, 100.0);
    Equation eq118_;
    eq118_.addAddend(-1, q118_);
    eq118_.addAddend(-0.0016654340025765723, variable);
    eq118_.setScalar(0.9876154660829327);
    _inputQuery.addEquation(eq118_);
    set4.insert(q118_);

    unsigned q_at_x_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(q_at_x_+1);
    //0.70448665747663167881338441808708
    _inputQuery.setLowerBound(q_at_x_, -1.0);
    _inputQuery.setUpperBound(q_at_x_, -1.0);
    set4.insert(q_at_x_);


    unsigned negative_q_at_x_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(negative_q_at_x_+1);
    _inputQuery.setLowerBound(negative_q_at_x_, -0.5);
    _inputQuery.setUpperBound(negative_q_at_x_, -0.5);
    printf("\nq_at_x_ = %d negative_q_at_x_ = %d", q_at_x_, negative_q_at_x_);

    unsigned zero_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(zero_+1);
    _inputQuery.setLowerBound(zero_, 0.0);
    _inputQuery.setUpperBound(zero_, 0.0);
    set3.insert(zero_);


    unsigned fmax3 = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(fmax3+1);
    _inputQuery.setLowerBound(fmax3,-100.0);
    _inputQuery.setUpperBound(fmax3,100.0);
    MaxConstraint *m3 = new MaxConstraint(fmax3, set3);
    _inputQuery.addPiecewiseLinearConstraint(m3);

    unsigned fmax4 = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(fmax4+1);
    _inputQuery.setLowerBound(fmax4,-100.0);
    _inputQuery.setUpperBound(fmax4,100.0);
    MaxConstraint *m4 = new MaxConstraint(fmax4, set4);
    _inputQuery.addPiecewiseLinearConstraint(m4);

    unsigned negative_fmax4 = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(negative_fmax4+1);
    _inputQuery.setLowerBound(negative_fmax4,-10.0);
    _inputQuery.setUpperBound(negative_fmax4,10.0);
    printf ("\n fmax3 = %d \tfmax4 = %d", fmax3,fmax4);
    Equation eq37_;
    eq37_.addAddend(1,fmax4);
    eq37_.addAddend(1,negative_fmax4);
    eq37_.setScalar(0);
    _inputQuery.addEquation(eq37_);

    unsigned negative_fmax3 = _inputQuery.getNumberOfVariables();
    printf("\tnegative_fmax1 = %u\t", negative_fmax3);
    _inputQuery.setNumberOfVariables(negative_fmax3+1);
    _inputQuery.setLowerBound(negative_fmax3,-100.0);
    _inputQuery.setUpperBound(negative_fmax3,100.0);
    Equation eq36_;
    eq36_.addAddend(1,fmax3);
    eq36_.addAddend(1,negative_fmax3);
    eq36_.setScalar(0);
    _inputQuery.addEquation(eq36_);

    unsigned variable_min3 = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(variable_min3+1);
    _inputQuery.setLowerBound(variable_min3, -100.0);
    _inputQuery.setUpperBound(variable_min3, 100.0);

    unsigned variable_min4 = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(variable_min4+1);
    _inputQuery.setLowerBound(variable_min4, -100.0);
    _inputQuery.setUpperBound(variable_min4, 100.0);

    unsigned point5 = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(point5+1);
    _inputQuery.setLowerBound(point5, 0.5);
    _inputQuery.setUpperBound(point5, 0.5);

    minSet3.insert(negative_q_at_x_);
    minSet3.insert(negative_fmax3);

    minSet4.insert(negative_fmax4);
    minSet4.insert(point5);

    unsigned neg_variable_min3 = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(neg_variable_min3+1);
    _inputQuery.setLowerBound(neg_variable_min3,-100.0);
    _inputQuery.setUpperBound(neg_variable_min3,100.0);

    unsigned neg_variable_min4 = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(neg_variable_min4+1);
    _inputQuery.setLowerBound(neg_variable_min4,-100.0);
    _inputQuery.setUpperBound(neg_variable_min4,100.0);

    MaxConstraint *min3 = new MaxConstraint(variable_min3, minSet3);
    _inputQuery.addPiecewiseLinearConstraint(min3);
    MaxConstraint *min4 = new MaxConstraint(variable_min4, minSet4);
    _inputQuery.addPiecewiseLinearConstraint(min4);

    printf("\n\nneg_variable_min3 = %d  neg_variable_min4 = %d\n\n", neg_variable_min3, neg_variable_min4);

    Equation eq38_;
    eq38_.addAddend(1,variable_min3);
    eq38_.addAddend(1,neg_variable_min3);
    eq38_.setScalar(0);
    _inputQuery.addEquation(eq38_);

    Equation eq39_;
    eq39_.addAddend(1,variable_min4);
    eq39_.addAddend(1,neg_variable_min4);
    eq39_.setScalar(0);
    _inputQuery.addEquation(eq39_);

    unsigned answer_ = _inputQuery.getNumberOfVariables();
    _inputQuery.setNumberOfVariables(answer_+1);
//    _inputQuery.setLowerBound(answer_,0.0);
//    _inputQuery.setUpperBound(answer_,1.0);
    Equation eq40_;
    eq40_.addAddend(1,answer_);
    eq40_.addAddend(-1,neg_variable_min3);
    eq40_.addAddend(-1,variable_min4);
    eq40_.setScalar(-0.5);
    _inputQuery.addEquation(eq40_);
    printf("\t\tanswer = %d\t\t", answer_);

    return answer_;
}
/**
 * Returns the natural logarithm for the integer n for n up to 10
 * @param n unsigned integer for which to produce the logarithm
 * @return the log value as double
 */
double log(unsigned n) {
    double x;
    if (n == 2) {
        x = 0.6931;
    } else if (n == 3) {
        x = 1.0986;
    } else if (n == 4) {
        x = 1.3863;
    } else if (n == 5) {
        x = 1.6094;
    } else if (n == 6) {
        x = 1.7917;
    } else if (n == 7) {
        x = 1.9459;
    } else if (n == 8) {
        x = 2.0794;
    } else if (n == 9) {
        x = 2.1972;
    } else if (n == 10) {
        x = 2.3025;
    } else {
        x = 0;
    }
    return x;
}

void Marabou::prepareInputQuery() {
    String inputQueryFilePath = Options::get()->getString(Options::INPUT_QUERY_FILE_PATH);
    if (inputQueryFilePath.length() > 0) {
        /*
          Step 1: extract the query
        */
        if (!File::exists(inputQueryFilePath)) {
            printf("Error: the specified inputQuery file (%s) doesn't exist!\n", inputQueryFilePath.ascii());
            throw MarabouError(MarabouError::FILE_DOESNT_EXIST, inputQueryFilePath.ascii());
        }

        printf("InputQuery: %s\n", inputQueryFilePath.ascii());
        _inputQuery = QueryLoader::loadQuery(inputQueryFilePath);

        // anagha: add max() and sigmoid() to _inputQuery here

        _inputQuery.constructNetworkLevelReasoner();
    } else {
        /*
          Step 1: extract the network
        */
        String networkFilePath = Options::get()->getString(Options::INPUT_FILE_PATH);
        if (!File::exists(networkFilePath)) {
            printf("Error: the specified network file (%s) doesn't exist!\n", networkFilePath.ascii());
            throw MarabouError(MarabouError::FILE_DOESNT_EXIST, networkFilePath.ascii());
        }
        printf("Network: %s\n", networkFilePath.ascii());

        // For now, assume the network is given in ACAS format
        _acasParser = new AcasParser(networkFilePath);
        _acasParser->generateQueryAnagha(_inputQuery);

        _inputQuery.constructNetworkLevelReasoner();
        _inputQuery.setLowerBound(3,-10);
        _inputQuery.setUpperBound(3,10);
        _inputQuery.setLowerBound(11,-10);
        _inputQuery.setUpperBound(11,10);

        List<unsigned> outlist1;
        List<unsigned> outlist2;

        unsigned outputLayerSize = _inputQuery.getNumOutputVariables();
        double result;
        result = log((outputLayerSize / 2) - 1);
        unsigned counter = 0;
        //TODO: what do id1 and id2 really do? why is it 2 lists independent of the number of classes?
        //ANSWER (assume): it is the two copies of the neural net!
        for (const auto &pair: _inputQuery._outputIndexToVariable) {
            if (counter < outputLayerSize / 2) {
                outlist1.append(pair.second);
                ++counter;
            } else {
                outlist2.append(pair.second);
                ++counter;
            }
        }

        //TODO my variables
        unsigned targetClass1, targetClass2;
        double conf_from_user, epsilon_from_user;
        String propertyFilePath = Options::get()->getString(Options::PROPERTY_FILE_PATH);
        if (propertyFilePath != "") {
            printf("Property: %s\n", propertyFilePath.ascii()); // called
            PropertyParser().parse(propertyFilePath, conf_from_user, epsilon_from_user, targetClass1, targetClass2);
            printf("conf_from_user = %f epsilon_from_user = %f", conf_from_user, epsilon_from_user);
        } else
            printf("Property: None\n");

        printf("\n");
        unsigned conf1;
        unsigned i = 0;
        for (const auto &outVar1: outlist1) {
            if (i++ != targetClass1) continue;//encode this stuff only for the interesting class

            Set<unsigned> maxSet1;
            for (const auto &outVar2: outlist1) {
                if (outVar1 == outVar2) continue;

                maxSet1.insert(outVar2);
                //make the targetclass the highest value.
                Equation predictRightClass(Equation::GE);
                predictRightClass.addAddend(1, outVar1);
                predictRightClass.addAddend(-1, outVar2);
                predictRightClass.setScalar(1e-6);
                _inputQuery.addEquation(predictRightClass);
            }



            //TODO: the softmax approximation is sigmoid(var_i - LSE (rest))
            // LSE(rest) is lower bounded by max(rest) ==> we upper bound confidence with sig(var_i - max(rest))
            // We can improve our estimate of LSE(rest) by adding a term like (max(rest),(max(rest)+log(2)-(2*max(rest)-rest1-rest2)/2))
            // two line segments, one we already had, one adds a line segment where we have the range of the rest as slope and log(2) as offset

            //(max(rest)+log(2)-(2*max(rest)-rest1-rest2)/2) = (log(2)+rest1/2+rest2/2
            // ==> log(n) +

            //TODO: LSE approximation:
            // LSE(x_1,x_2,x_3,...) >= max(x) original approx
            // new approach, consider LSE(x_1-x_min,x_2-x_min,...)+x_min = LSE(x_i)
            // We have LSE(x_1-x_min,... 0, x_j-x_min,...) now, so one less degree of freedom.

            //TODO special case for 2 numbers (three classes)
            // For 2 numbers, we now have only the function LSE(x1,x2) = LSE(x_max-x_min,0)+x_min = log(e^(x_max-x_min)+e^0)+x_min = f(x_r)+x_min
            // We approximate this function with:
            // f(x_r) >= x_r (non-negative)
            // f(x_r) >= log(2) + x_r/2 (slope at f(0)
            // ==>
            // f(x_r) >= max(x_r,log(2)-x_r/2) = max(x_r,log(2)+x_r/2)
            // f(x_r) + x_min >= max(x_r + x_min,log(2)+x_r/2 + x_min)
            // f(x_r) + x_min >= max(x_max,log(2)+x_max/2 + x_min/2)
            // f(x_r) + x_min >= max(x_max,log(2)+x_max/2 + (x1 + x2 - x_max)/2)
            // f(x_r) + x_min >= max(x_max,log(2)+x_max/2 + x1/2 + x2/2 - x_max/2)
            // f(x_r) + x_min >= max(x_max,log(2) + x1/2 + x2/2)

            // x_r is the range x_max - x_min (for two numbers: 2*x_max - x_1 - x_2)

            unsigned temp_conf;

            if(outlist1.size() == 3){
                //TODO: encode the variable lse_taylor1 = log(2)+0.5*x_1+0.5*x_2
                // hint: if only one x is left, we have log(0) + .5*x, so we only want to add this for three variables
                // reformulate to lse_taylor1 - .5*x_1 - .5*x_2 <= log(2)
                unsigned lse_taylor1 = _inputQuery.getNumberOfVariables();
                _inputQuery.setNumberOfVariables(lse_taylor1 + 1);
                _inputQuery.setUpperBound(lse_taylor1, 1000.0);
                _inputQuery.setLowerBound(lse_taylor1, -1000.0);
                Equation lse_taylor_approximation(Equation::LE);
                //TODO: we relax this constraint to >= and <= constraints, otherwise marabou dies
                lse_taylor_approximation.addAddend(1, lse_taylor1);
                for (const auto &outVar2: outlist1)
                    if (&outVar1 != &outVar2)
                        lse_taylor_approximation.addAddend(-0.5, outVar2);
                lse_taylor_approximation.setScalar(std::log(2));
                _inputQuery.addEquation(lse_taylor_approximation);

                Equation lse_taylor_approximation2(Equation::GE);
                //TODO: we relax this constraint to allow smaller values, otherwise marabou dies
                lse_taylor_approximation2.addAddend(1, lse_taylor1);
                for (const auto &outVar2: outlist1)
                    if (&outVar1 != &outVar2)
                        lse_taylor_approximation2.addAddend(-0.5, outVar2);
                lse_taylor_approximation2.setScalar(std::log(2));
                _inputQuery.addEquation(lse_taylor_approximation2);

                maxSet1.insert(lse_taylor1);
                unsigned lse_max = _inputQuery.getNumberOfVariables();//TODO marabou does not like this constraint!
                _inputQuery.setNumberOfVariables(lse_max + 1);
                _inputQuery.setUpperBound(lse_max, 1000.0);
                _inputQuery.setLowerBound(lse_max, -1000.0);
                auto *max_lse_constraint = new MaxConstraint(lse_max, maxSet1);
                _inputQuery.addPiecewiseLinearConstraint(max_lse_constraint);
                printf("\napproximated lse (for 3 classes) = %d", lse_max);

                unsigned sigmoid_input = _inputQuery.getNumberOfVariables();
                _inputQuery.setNumberOfVariables(sigmoid_input + 1);
                _inputQuery.setUpperBound(sigmoid_input, 1000.0);
                _inputQuery.setLowerBound(sigmoid_input, -1000.0);

                Equation equation3;
                equation3.addAddend(+1, lse_max);
                equation3.addAddend(-1, outVar1);
                equation3.addAddend(1, sigmoid_input); //lse_approx - outVar1 + max_var1 = log(output_dimensions)
                equation3.setScalar(0);
                _inputQuery.addEquation(equation3);
                temp_conf = sigmoid_anagha_final(sigmoid_input);//redefined, new number of variables (or number or last one)
            }else{

                unsigned max_var1 = _inputQuery.getNumberOfVariables();
                _inputQuery.setNumberOfVariables(max_var1 + 1);
                _inputQuery.setUpperBound(max_var1, 1000.0);
                _inputQuery.setLowerBound(max_var1, -1000.0); //TODO: what is the meaning of this?
                auto *max1 = new MaxConstraint(max_var1, maxSet1);//variable number max_var1 is the maximum of all other outputs
                _inputQuery.addPiecewiseLinearConstraint(max1);//TODO: can be bounded with the range of the variables

                printf("\nmax input of other classes = %d", max_var1);
                signed lse_approx = _inputQuery.getNumberOfVariables();
                _inputQuery.setNumberOfVariables(lse_approx + 1);
                _inputQuery.setUpperBound(lse_approx, 1000.0);
                _inputQuery.setLowerBound(lse_approx, -1000.0); //TODO: what is the meaning of this?
                Equation equation2;
                equation2.addAddend(1, lse_approx); //lse_approx - outVar1 + max_var1 = log(output_dimensions)
                equation2.addAddend(-1, outVar1); // ==> lse_approx = log(output_dimensions) + outVar1 - max_var1
                equation2.addAddend(+1, max_var1); //TODO: still dont get it, probably something with the softmax stuff
                equation2.setScalar(0);//TODO can be bounded with bounds of variables
                _inputQuery.addEquation(equation2);
                printf("\napproximated lse = %d", lse_approx);
                temp_conf = sigmoid_anagha_final(lse_approx);//redefined, new number of variables (or number or last one)
            }
//            if(i++ != targetClass1) continue;
            conf1 = temp_conf;
            printf("\nconfidence = %d", conf1);

        }

        i = 0;
        for (const auto &outVar1: outlist2) {
            if (i++ != targetClass2) continue;//encode this stuff only for the interesting class

            for (const auto &outVar2: outlist2) {
                if (&outVar1 != &outVar2) {
                    //make the targetclass the highest value.
                    Equation predictRightClass(Equation::GE);
                    predictRightClass.addAddend(1, outVar1);
                    predictRightClass.addAddend(-1, outVar2);
                    predictRightClass.setScalar(1e-6);
                    _inputQuery.addEquation(predictRightClass);
                }
            }
        }

        /*
          Step 2: extract the property in question
        */

        unsigned counterX = 0;
        unsigned counterInVar = _inputQuery.getNumInputVariables() / 2;//TODO the variable directly after the inputs? not used? the first variable in the copy?
//        _inputQuery.setUpperBound(counterInVar, 10000.0);
//        _inputQuery.setLowerBound(counterInVar, -10000.0);

        //TODO: ALTERNATIVE CODE SNIPPET FOR THE LOOP STARTING HERE
        while(counterX < (_inputQuery.getNumInputVariables())/2)
        {

            Equation distanceMin(Equation::LE);
            distanceMin.addAddend(-1,counterX);
            distanceMin.addAddend(1,counterX + counterInVar);
            distanceMin.setScalar(epsilon_from_user);
            //TODO: de-normalize epsilon: distanceMax.setScalar(epsilon_from_user/(_inputQuery.getUpperBound(counterX)-_inputQuery.getLowerBound(counterX)));
            _inputQuery.addEquation(distanceMin);

            Equation distanceMax(Equation::LE);
            distanceMax.addAddend(1,counterX);
            distanceMax.addAddend(-1,counterX + counterInVar);
            distanceMax.setScalar(epsilon_from_user);
            //TODO: de-normalize epsilon: distanceMax.setScalar(epsilon_from_user/(_inputQuery.getUpperBound(counterX)-_inputQuery.getLowerBound(counterX)));
            _inputQuery.addEquation(distanceMax);

            ++counterX;
        }
        //TODO:ALTERNATIVE CODE SNIPPET ENDING HERE

        /*Property starts here*/
        //TODO new correction from PETER:
        double conf_from_user_app = conf_from_user;// - 0.0556; //TODO: apparent correction for 3 class confidence issue
        Equation confidenceThreshold(Equation::GE);
        confidenceThreshold.addAddend(1, conf1);
        confidenceThreshold.setScalar(conf_from_user_app);
        _inputQuery.addEquation(confidenceThreshold);//TODO: max confidence >= user defined confidence



    }
    sleep(5);
    if (Options::get()->getBool(Options::DEBUG_ASSIGNMENT))
        importDebuggingSolution();

    String queryDumpFilePath = Options::get()->getString(Options::QUERY_DUMP_FILE);
    if (queryDumpFilePath.length() > 0) {
        _inputQuery.saveQuery(queryDumpFilePath);
        printf("\nInput query successfully dumped to file\n");
        exit(0);
    }
}

void Marabou::importDebuggingSolution()
{
    String fileName = Options::get()->getString(Options::IMPORT_ASSIGNMENT_FILE_PATH);
    AutoFile input(fileName);

    if (!IFile::exists(fileName))
    {
        throw MarabouError(MarabouError::FILE_DOES_NOT_EXIST, Stringf("File %s not found.\n", fileName.ascii()).ascii());
    }

    input->open(IFile::MODE_READ);

    unsigned numVars = atoi(input->readLine().trim().ascii());
    ASSERT(numVars == _inputQuery.getNumberOfVariables());

    unsigned var;
    double value;
    String line;

    // Import each assignment
    for (unsigned i = 0; i < numVars; ++i)
    {
        line = input->readLine();
        List<String> tokens = line.tokenize(",");
        auto it = tokens.begin();
        var = atoi(it->ascii());
        ASSERT(var == i);
        it++;
        value = atof(it->ascii());
        it++;
        ASSERT(it == tokens.end());
        _inputQuery.storeDebuggingSolution(var, value);
    }

    input->close();
}

void Marabou::exportAssignment() const
{
    String assignmentFileName = "assignment.txt";
    AutoFile exportFile(assignmentFileName);
    exportFile->open(IFile::MODE_WRITE_TRUNCATE);

    unsigned numberOfVariables = _inputQuery.getNumberOfVariables();
    // Number of Variables
    exportFile->write(Stringf("%u\n", numberOfVariables));

    // Export each assignment
    for (unsigned var = 0; var < numberOfVariables; ++var)
        exportFile->write(Stringf("%u, %f\n", var, _inputQuery.getSolutionValue(var)));

    exportFile->close();
}

void Marabou::solveQuery()
{
    if (_engine.processInputQuery(_inputQuery))
        _engine.solve(Options::get()->getInt(Options::TIMEOUT));

    if (_engine.getExitCode() == Engine::SAT)
        _engine.extractSolution(_inputQuery);
}

double rescaleToOriginal(double x, double originalMin, double originalMax, double scaledMin, double scaledMax) {
    return ((x - scaledMin) * (originalMax - originalMin)) / (scaledMax - scaledMin) + originalMin;
}

void Marabou::displayResults(unsigned long long microSecondsElapsed) const
{
    Engine::ExitCode result = _engine.getExitCode();
    String resultString;

    if (result == Engine::UNSAT)
    {
        resultString = "unsat";
        printf("unsat :)\n");
    }
    else if (result == Engine::SAT)
    {
        resultString = "sat";
        printf("sat\n");

        printf("\n\nRaw Input assignment for _inputQuery:\n\n");
        for (unsigned i = 0; i < _inputQuery.getNumInputVariables(); ++i){
            double min,max;
            _acasParser->getRawInputRange(_inputQuery.inputVariableByIndex(i % (_inputQuery.getNumInputVariables()/2)), min, max);
            double smin = _inputQuery.getLowerBound(_inputQuery.inputVariableByIndex(i));
            double smax = _inputQuery.getUpperBound(_inputQuery.inputVariableByIndex(i));
            if(i == _inputQuery.getNumInputVariables()/2)printf("\n");
            printf("\tx%u = %lf [%lf,%lf]\n", i,
                   rescaleToOriginal(_inputQuery.getSolutionValue(_inputQuery.inputVariableByIndex(i)),min,max,smin,smax),
                   min,
                   max);

        }
        printf("\n\tApproximate Confidence at input 1 (last variable in model): %lf\n",
               _inputQuery.getSolutionValue(
                               _inputQuery.getNumberOfVariables()-1));
        printf("\tMinimum possible confidence at input 1 (with correction for 3 outputs): %lf\n",
               _inputQuery.getSolutionValue(_inputQuery.getNumberOfVariables()-1)-0.0556);
        printf("\n\tRaw Output values (always 6 variables outputted):\n");
        for (unsigned output: _inputQuery.getOutputVariables()){
            printf("\tx%u = %lf\n",output,_inputQuery.getSolutionValue(output));
        }


        if (_inputQuery._networkLevelReasoner)
        {
            double *input = new double[_inputQuery.getNumInputVariables()];
            for (unsigned i = 0; i < _inputQuery.getNumInputVariables(); ++i)
                input[i] = _inputQuery.getSolutionValue(_inputQuery.inputVariableByIndex(i));


            for(unsigned i = 0; i < _inputQuery.getNumberOfVariables(); i++)
            {
                double xy = _inputQuery.getSolutionValue(i);
                printf("\nSolution value x%u = %f", i, xy);
            }

            NLR::NetworkLevelReasoner *nlr = _inputQuery._networkLevelReasoner;
            NLR::Layer *lastLayer = nlr->getLayer(nlr->getNumberOfLayers() - 1);
            double *output = new double[lastLayer->getSize()];

            nlr->evaluate(input, output);

            /*printf("\n");
            printf("Output:\n");
            for (unsigned i = 0; i < lastLayer->getSize(); ++i)
                printf("\ty%u = %lf\n", i, output[i]);
            printf("\n");*/
            delete[] input;
            delete[] output;
        }
        else
        {
            /*printf("\n");
            printf("Output:\n");
            for (unsigned i = 0; i < _inputQuery.getNumOutputVariables(); ++i)
                printf("\ty%u = %lf\n", i, _inputQuery.getSolutionValue(_inputQuery.outputVariableByIndex(i)));*/
            printf("\n");
        }
    }
    else if (result == Engine::TIMEOUT)
    {
        resultString = "TIMEOUT";
        printf("Timeout\n");
    }
    else if (result == Engine::ERROR)
    {
        resultString = "ERROR";
        printf("Error\n");
    }
    else
    {
        resultString = "UNKNOWN";
        printf("UNKNOWN EXIT CODE! (this should not happen)");
    }

    // Create a summary file, if requested
    String summaryFilePath = Options::get()->getString(Options::SUMMARY_FILE);
    if (summaryFilePath != "")
    {
        File summaryFile(summaryFilePath);
        summaryFile.open(File::MODE_WRITE_TRUNCATE);

        // Field #1: result
        summaryFile.write(resultString);

        // Field #2: total elapsed time
        summaryFile.write(Stringf(" %u ", microSecondsElapsed / 1000000)); // In seconds

        // Field #3: number of visited tree states
        summaryFile.write(Stringf("%u ",
                                  _engine.getStatistics()->getUnsignedAttribute(Statistics::NUM_VISITED_TREE_STATES)));

        // Field #4: average pivot time in micro seconds
        summaryFile.write(Stringf("%u",
                                  _engine.getStatistics()->getAveragePivotTimeInMicro()));

        summaryFile.write("\n");
    }
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
