Title: Performance analysis of Machine Learning (ML) loads.

Duration: 1 day

Exercise: 
A team developing ML devices has requested some analysis be done on the performance data produced by their testing infrastructure. 
The data is contained within two files (TestInfo.csv & TestResults.pickle) and are provided to you. 
You are tasked with analysing the performance data and providing a summary of your findings in an executive summary, a clear
visualisation of the findings, and suggestions related to the findings.


Deliverables: 
1) Executive summary of your findings (A clear overview of the findings summarized on one page, understandable for an audience that 
have not seen the data or have a detailed understanding of it).
2) Visualisations that helped you analyse the data with added annotations. 
3) Follow up actions regarding your findings you would take with the development team 

Guidance and hints: 
- The data within the two files are related to each other by the field 'TestId'.
- TestResults.pickle was generated using pandas and Python2.7. 
- The pickle file can be loaded using Python3 with the parameter encoding='latin1'
- This is not a quantitative analysis so there is no need to do statistical analysis of the data.
- As the data is already there the focus is on understanding the data and be able to clearly communicate and visualise the findings. 
- Please include any reasons/justifications in your summary as to why the observed behaviour may have occurred.
- The testID is used only to map tests and results.
- Build can be interpreted as different versions of the SW.
- Device can be interpreted as different versions of HW.

