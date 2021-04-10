#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt 
import emoji
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
import copy
import liwc
from collections import Counter


# In[2]:


"""
This is for Gale! To be used for analysis paper at ACII 2021.

This is still datapoint level analysis. We will not worry about individual participants being repeated or not.
We directly look at the participants associated with each data point.

Get everything in 1 file.
Variables are of two types: individual and diadic.
There is another categorization: distal, proximal, outcome variables.
Include everything. Use filters in SPSS for analysis.

In total, the data should have 1030*2 = 2060 rows.
Need to also create a document of all variables for Gale.
"""
print()


# In[3]:


in_f = "../storage/airtable/all_data-1030-shuffled.json" 
with open(in_f) as f:
    all_data = json.load(f)
    
print(len(all_data))
print(all_data[0].keys())


# In[4]:


def get_dialog_ann_data(dialog):
    #print(dialog)
    assert len(dialog) > 0
    did = dialog[0][0]
    for item in dialog:
        assert did == item[0]
        
    info = {
        'agent_1': {},
        'agent_2': {}
    }
    
    for item in dialog:
        aid = item[1]
        labels = item[2].split(",")
        for label in labels:
            if(label not in info[aid]):
                info[aid][label] = 0
            info[aid][label] += 1
    
    return info

def get_ann_data(input_files):
    """
    dict from dialogue id to all the info of that dialogue.
    """
    did2info = {}
    
    extra_utterances = ['Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away', 'Submit-Post-Survey']
    
    for fname in input_files:
        print(fname)
        df = pd.read_csv(fname)
        dat = pd.DataFrame.to_dict(df, orient="records")
        
        cur_dialog = []
        cur_id = -1
        
        for i, item in enumerate(dat):
            if((item["DialogueId"] != item["DialogueId"]) or (item["Utterance"] in extra_utterances)):
                continue
            
            #valid item
            this_id = item["DialogueId"]
            if((this_id != cur_id) and cur_dialog):
                #found new id
                cur_data = get_dialog_ann_data(cur_dialog[:])
                did2info[int(cur_id)] = cur_data
                cur_dialog = []

            if(isinstance(item["Labels"], str)):
                assert isinstance(item["Labels"], str) and len(item["Labels"])>0, i
                this_item = [item["DialogueId"], item["AgentID"], item["Labels"]]
                cur_dialog.append(this_item)
                cur_id = this_id

        if(cur_dialog):
            cur_data = get_dialog_ann_data(cur_dialog[:])
            did2info[int(cur_id)] = cur_data
            cur_dialog = []
            
    return did2info
    
input_files = [
    "../storage/airtable/accumulate/sample-5-kushal.csv",
    "../storage/airtable/accumulate/sample-200-kushal-Grid view-v2.csv",
    "../storage/airtable/accumulate/sample-200-rene-Grid view.csv",
    "../storage/airtable/accumulate/sample-200-jaysa-Grid view.csv",
    "../storage/airtable/accumulate/sample-next-10-kushal-Grid view_v2.csv"
]

did2info = get_ann_data(input_files)


# In[6]:


#liwc setup
parse, category_names = liwc.load_token_parser('/home/ICT2000/chawla/Documents/work/CSCI662/project-EMNLP2020/main/resources/LIWC2015_English.dic')
category_names = sorted(category_names)

def get_wid_liwc_data(wid, item):
    
    wid_info = {}
    
    all_text = ""
    for act in item["acts"]:
        if((act['id'] == wid) and (act['text'] not in ['Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away', 'Submit-Post-Survey'])):
            all_text += " " + act['text']
            
    tokens = [w.lower() for w in nltk.word_tokenize(all_text)]
    feat_dict = dict(Counter(category for token in tokens for category in parse(token)))
    
    for fname in category_names:
        fval = 0
        
        if(fname in feat_dict):
            fval = feat_dict[fname]

        wid_info["LIWC-" + fname.replace(" ", "-").replace("(", "").replace(")", "")] = fval
        
    return wid_info

def get_dialog_liwc_data(dialog):
    
    info = {
        'agent_1': get_wid_liwc_data("mturk_agent_1", dialog),
        'agent_2': get_wid_liwc_data("mturk_agent_2", dialog)
    }
    
    return info

def get_liwc_data():
    """
    dict from dialogue id to all the info of that dialogue.
    """
    
    did2liwcinfo = {}
    
    for item in all_data:
        
        did = item['dialogue_id']
        did2liwcinfo[did] = get_dialog_liwc_data(item)
    
    return did2liwcinfo

did2liwcinfo = get_liwc_data()


# In[24]:


def get_ann_counts(did, wid, label):
    
    key = 'agent_1'
    if('agent_2' in wid):
        key = 'agent_2'
    
    if(label in did2info[did][key]):
        return did2info[did][key][label]
    
    return 0

def get_individual_annotation(item, wid):
    """
    look at the dialogue, get the id, and extract the annotation counts for wid user from that id.
    """
    labels = ['Small-talk', 'Required-self', 'Required-other', 'Not-Required', 'Preference-elicitation', 
              'Undervalue-Other-Requirement', 'Vouching-for-fairness']
    
    did = item['dialogue_id']
    
    ann_counts = {}
    
    if(did in did2info):
        for label in labels:
            ann_counts["Strategy-" + label] = get_ann_counts(did, wid, label)
    else:
        for label in labels:
            ann_counts["Strategy-" + label] = -1

    return ann_counts

def get_liwc_counts(did, wid, label):
    
    key = 'agent_1'
    if('agent_2' in wid):
        key = 'agent_2'
    
    assert label in did2liwcinfo[did][key]
    return did2liwcinfo[did][key][label]
    
def get_individual_liwc(item, wid):
    """
    look at the dialogue, get the id, and extract the annotation counts for wid user from that id.
    """
    
    did = item['dialogue_id']
    
    liwc_counts = {}
    
    for fname in category_names:
        label = "LIWC-" + fname.replace(" ", "-").replace("(", "").replace(")", "")
        liwc_counts[label] = get_liwc_counts(did, wid, label)
            
    return liwc_counts

def get_worker_qualtrics(item, wid):
    
    for worker in item["workers"]:
        if(worker["id"] == wid):
            return item["qualtrics"][worker["worker_id"]]

def get_education_score(qualtrics):
    
    key2score = {
        'Some high school, no diploma': 1,
        'High school graduate / GED': 2,
        'Some 2 year college, no degree': 3,
        "Some 2 year college, associate's degree": 4,
        "Some 4 year college, no degree": 5,
        "Some 4 year college, bachelor's degree": 6,
        "Master's degree": 7,
        "Doctorate degree": 8,
        "Trade school": -1,
    }
    
    return key2score[qualtrics['Education']]

def get_SVO_label(qualtrics):
    """
    https://static1.squarespace.com/static/523f28fce4b0f99c83f055f2/t/56c794cdf8baf3ae17cf188c/1455920333224/Triple+Dominance+Measure+of+SVO.pdf
    """
    SVOans = [qualtrics["SVO " + str(i)] for i in range(1, 13)]
    
    prosocial = set(["You get: 480, Other gets 480", "You get: 490, Other gets 490", "You get: 500, Other gets 500", 
                "You get: 510, Other gets 510", "You get: 520, Other gets 520", "You get: 440, Other gets 440"])

    individualistic = set(["You get: 520, Other gets 300", "You get: 560, Other gets 300", "You get: 570, Other gets 300",
                      "You get: 550, Other gets 300", "You get: 530, Other gets 320", "You get: 580, Other gets 320",
                      "You get: 540, Other gets 300", "You get: 470, Other gets 300", "You get: 540, Other gets 280"])

    competitive = set(["You get: 480, Other gets 180", "You get: 500, Other gets 100", "You get: 460, Other gets 100",
                  "You get: 520, Other gets 120", "You get: 480, Other gets 100", "You get: 510, Other gets 110",
                  "You get: 330, Other gets 110", "You get: 490, Other gets 90", "You get: 480, Other gets 80"])
    
    all_choices = prosocial|individualistic|competitive
    
    pCount, iCount, cCount = 0, 0, 0
    
    for ans in SVOans:
        assert ans in all_choices, ans
        
        if(ans in prosocial):
            pCount += 1
            if(pCount > 6):
                return "Prosocial"
        elif(ans in individualistic):
            iCount += 1
            if(iCount > 6):
                return "Proself"
                #return "Individualistic"
        elif(ans in competitive):
            cCount += 1
            if(cCount > 6):
                return "Proself"
                #return "Competitive"
    
    #no category reached 6
    return "Unclassified"

def get_big_five_scores(qualtrics):
    """
    https://gosling.psy.utexas.edu/scales-weve-developed/ten-item-personality-measure-tipi/ten-item-personality-inventory-tipi/
    https://www.psychologytoday.com/us/blog/darwins-subterranean-world/201810/take-quick-personality-test
    """
    
    bigFiveAns = [qualtrics["Big Five_" + str(i)] for i in range(1,11)]
    bigFiveQues = ['Extraverted, enthusiastic', 'Critical, quarrelsome', 'Dependable, self-disciplined',
                    'Anxious, easily upset', 'Open to new experiences, complex', 'Reserved, quiet',
                    'Sympathetic, warm', 'Disorganized, careless', 'Calm, emotionally stable', 
                    'Conventional, uncreative']
    
    numTitles = {"Disagree Strongly":1, "Disagree Moderately":2, "Disagree a little":3, "Neither agree or disagree":4,
            "Agree a little":5, "Agree Moderately":6, "Agree Strongly":7}
    
    BB = {}
    for ques, ans in zip(bigFiveQues, bigFiveAns):
        BB[ques] = numTitles[ans]
        
    BB["Critical, quarrelsome"] = 8 - BB["Critical, quarrelsome"]    
    BB["Anxious, easily upset"] = 8 - BB["Anxious, easily upset"]     
    BB["Reserved, quiet"] = 8 - BB["Reserved, quiet"]            
    BB["Disorganized, careless"] = 8 - BB["Disorganized, careless"]  
    BB["Conventional, uncreative"] = 8 - BB["Conventional, uncreative"]
    
    big_five_scores = {
    
    "BigFive-Extraversion": (BB["Extraverted, enthusiastic"] + BB["Reserved, quiet"])/2,
    "BigFive-Agreeableness": (BB["Critical, quarrelsome"] + BB["Sympathetic, warm"])/2,
    "BigFive-Conscientiousness": (BB["Dependable, self-disciplined"] + BB["Disorganized, careless"])/2,
    "BigFive-Emotional-Stability": (BB["Anxious, easily upset"] + BB["Calm, emotionally stable"])/2,
    "BigFive-Openness-to-Experiences": (BB["Open to new experiences, complex"] + BB["Conventional, uncreative"])/2
    
    }
    
    return big_five_scores

def get_individual_distal(item, wid):
    """
    This is the qualtrics stuff.
    """
    qualtrics = get_worker_qualtrics(item, wid)
    
    usable_data = {}
    
    usable_data["Age"] = int(qualtrics['Age'])
    usable_data["Gender"] = qualtrics['Gender']
    usable_data["Ethnicity"] = qualtrics["Race/Ethnicity"]
    
    usable_data["Education"] = get_education_score(qualtrics)
    usable_data["SVO"] = get_SVO_label(qualtrics)
    
    big_five_scores = get_big_five_scores(qualtrics)
    for attribute, score in big_five_scores.items():
        usable_data[attribute] = score
        
    return usable_data

def get_emoticon_counts(text):
    """
    '🙂', '☹️', '😮', '😡'
    """
    counts = {
        'Num-happy': text.count('🙂'),
        'Num-sad': text.count('☹️'),
        'Num-surprise': text.count('😮'),
        'Num-angry': text.count('😡')
    }
    
    return counts

def get_individual_proximal(item, wid):
    """
    These are various features extracted from the conversation and negotiation scenario (proximal)
    goes_first, num_words, num_emoticon_anger//etc, walks_away, high_issue, medium_issue, low_issue, 
    reason category (can you try with simple keywords?) do like topic modelling on the reasons (yesssss)
    """
    individual_proximal_info = {}
    
    for worker in item["workers"]:
        if(worker['id'] == wid):
            individual_proximal_info['High_item'] = worker['value2issue']['High']
            individual_proximal_info['Medium_item'] = worker['value2issue']['Medium']
            individual_proximal_info['Low_item'] = worker['value2issue']['Low']
            
    for act in item["acts"]:
        if(act['text'] not in ['Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away', 'Submit-Post-Survey']):
            if(act['id'] == wid):
                individual_proximal_info["Goes_first"] = "yes"
            else:
                individual_proximal_info["Goes_first"] = "no"
            break
            
    individual_proximal_info["Walks_away"] = "no"
    for act in item["acts"]:
        if(act['text'] == 'Walk-Away'):
            if(act['id'] == wid):
                individual_proximal_info["Walks_away"] = "yes"
            break
    
    all_text = ""
    for act in item["acts"]:
        if((act['id'] == wid) and (act['text'] not in ['Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away', 'Submit-Post-Survey'])):
            all_text += " " + act['text']
    
    individual_proximal_info["Num_words"] = len(nltk.word_tokenize(all_text))
    
    #emoticons
    individual_proximal_info.update(get_emoticon_counts(all_text))
    
    return individual_proximal_info

def get_satisfaction(sat_type):
    type2score = {
        'Extremely dissatisfied': 1,
        'Slightly dissatisfied': 2,
        'Undecided': 3,
        'Slightly satisfied': 4,
        'Extremely satisfied': 5,
    }
    
    return type2score[sat_type]

def get_opp_likeness(like_type):
    type2score = {
        'Extremely dislike': 1,
        'Slightly dislike': 2,
        'Undecided': 3,
        'Slightly like': 4,
        'Extremely like': 5,
    }
    
    return type2score[like_type]

def get_partner_issue(item, wid, value):
    
    for worker in item['workers']:
        if(worker['id'] != wid):
            return worker['value2issue'][value]

def get_individual_outcomes(item, wid):
    """
    points scored, satisfaction, likeness, guess partner high, guess partner low.
    H:5, M:4, L:3
    """
    individual_outcomes = {}
    
    #points scored.
    someone_walks_away = False
    for act in item["acts"]:
        if(act['text'] == 'Walk-Away'):
            someone_walks_away = True
            break
    
    if(someone_walks_away):
        individual_outcomes['Points_scored'] = 5#=one high item
    else:
        assert item['acts'][-4]['text'] == 'Submit-Deal'
        #this is the final deal.
        deal = item['acts'][-4]
        
        if(deal['id'] == wid):
            #this worker submitted the deal.
            key = 'issue2youget'
        else:
            #this worker did not submit the deal, so this worker gets whatever they gets
            key = 'issue2theyget'
            
        #get value2issue
        for worker in item['workers']:
            if(worker['id'] == wid):
                value2issue = worker['value2issue']
                break
                
        points = 0
        points += 5*int(deal['task_data']['response'][key][value2issue['High']])
        points += 4*int(deal['task_data']['response'][key][value2issue['Medium']])
        points += 3*int(deal['task_data']['response'][key][value2issue['Low']])
        
        individual_outcomes['Points_scored'] = points
        
    #from the post-survey
    for act in item['acts']:
        if(act['text'] == 'Submit-Post-Survey' and (act['id'] == wid)):
            individual_outcomes['Satisfaction'] = get_satisfaction(act['task_data']['response']['satisfaction'])
            individual_outcomes['Opponent_likeness'] = get_opp_likeness(act['task_data']['response']['likeness'])
            
            if(get_partner_issue(item, wid, "High") == act['task_data']['response']['partner_highest_item']):
                individual_outcomes['Guess_partner_high'] = 'Correct'
            else:
                individual_outcomes['Guess_partner_high'] = 'Incorrect'
            
            if(get_partner_issue(item, wid, "Low") == act['task_data']['response']['partner_lowest_item']):
                individual_outcomes['Guess_partner_low'] = 'Correct'
            else:
                individual_outcomes['Guess_partner_low'] = 'Incorrect'
                
    return individual_outcomes

def get_individual_info(item, wid):
    """
    individual level information, from all the ends, distal, proximal and outcomes.
    """
    individual_info = {}
    individual_info["Wid"] = wid
    
    individual_info.update(get_individual_distal(item, wid))
    individual_info.update(get_individual_proximal(item, wid))
    individual_info.update(get_individual_annotation(item, wid))
    individual_info.update(get_individual_liwc(item, wid))
    individual_info.update(get_individual_outcomes(item, wid))
    
    return individual_info

def get_integrative_potential(individuals_info):
    """
    scenario type
    
    only 3 categories.
    
    Integrative Potential:
    HH, MM, LL: max score 36, Code: 1
    HM, HM, LL/ HH, ML, ML: 39 Code: 2,
    HM, ML, HL/HL, MM, HL: 42 Code: 3
    
    Integrative potential increases with increasing code number. -> continuous.
    """
    
    if((individuals_info[0]['Low_item'] == individuals_info[1]['Low_item']) and (individuals_info[0]['Medium_item'] == individuals_info[1]['Medium_item'])):
        return 1
    
    if((individuals_info[0]['Low_item'] == individuals_info[1]['Low_item']) and (individuals_info[0]['High_item'] == individuals_info[1]['Medium_item'])):
        return 2
    
    if((individuals_info[0]['High_item'] == individuals_info[1]['High_item']) and (individuals_info[0]['Medium_item'] == individuals_info[1]['Low_item'])):
        return 2
    
    return 3

def get_diadic_info(item, index, individuals_info):
    """
    combining continuous, categorical individual variables, 
    scenario type.
    """
    diadic_info = {}
    
    diadic_info["Conversation-id"] = index
    
    #scenario type
    diadic_info["Integrative-potential"] = get_integrative_potential(individuals_info)
    
    #Walk away
    dkey = "Someone-Walks-Away"
    ikey = "Walks_away"
    if(individuals_info[0][ikey] == "yes" or individuals_info[1][ikey] == "yes"):
        diadic_info[dkey] = "yes"
    else:
        diadic_info[dkey] = "no"
        
    #total points scored.
    dkey = "Total-points-scored"
    ikey = "Points_scored"
    diadic_info[dkey] = individuals_info[0][ikey] + individuals_info[1][ikey]
    
    return diadic_info

def intermix_individuals_info(individuals_info):
    individuals_info_mixed = [copy.deepcopy(ii) for ii in individuals_info]
    
    for i in range(2):
        partner_i = 1-i
        for key, value in individuals_info[partner_i].items():
            individuals_info_mixed[i]['Partner.' + key] = value
            
    return individuals_info_mixed

def get_gale_point_info(item, index):
    
    ids = ["mturk_agent_1", "mturk_agent_2"]
    #individual info
    individuals_info = [get_individual_info(item, wid) for wid in ids]
    
    #diadic variables
    diadic_info = get_diadic_info(item, index, individuals_info)
    
    #intermix: Use a prefix like Partner. for partner's info.
    individuals_info_mixed = intermix_individuals_info(individuals_info)
    
    #add diadic.
    for i in range(2):
        individuals_info_mixed[i].update(diadic_info)
        
    #once inter-mixed + diadic, becomes the final datapoint
    gale_point_info = individuals_info_mixed
    return gale_point_info


# In[27]:


gale_level_info = [] # list of dicts
for ix, item in enumerate(all_data, start=1):
    
    #list of size two, one for each participant, add annos whenever available, otherwise keeep None.
    this_info = get_gale_point_info(item, ix)
    gale_level_info += this_info
        
df = pd.DataFrame.from_dict(gale_level_info)
df.to_csv('../storage/gale-level-info.csv', index=False)


# In[28]:


print(df.columns.tolist())


# In[ ]:




