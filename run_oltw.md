# **OLTW-Global Execution Instructions**

## **Prerequisites**

1. **Java:** java command must be available in PATH.  
2. **Library:** pip install OnlineAlignment  
3. **Files:** PerformanceMatcher.jar, query\_audio.wav, reference\_audio.wav.

## **Execution Methods**

### **1\. Python API**

from OnlineAlignment import run\_offline\_oltw  
\# Call run\_offline\_oltw with required audio paths

### **2\. Subprocess (CLI Wrapper)**

import subprocess

jar\_path \= "PerformanceMatcher.jar"  
query\_path \= "query.wav"  
reference\_path \= "ref.wav"  
output\_path \= "alignment.txt"

cmd \= \[  
    'java', '-jar', jar\_path,  
    '-b', '-q', '-G', '-D', '--use-chroma-map',  
    query\_path, reference\_path  
\]

with open(output\_path, 'w') as f:  
    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)

### **3\. Terminal Command**

java \-jar PerformanceMatcher.jar \-b \-q \-G \-D \--use-chroma-map \[query\_path\] \[reference\_path\] \> alignment\_output.txt

## **Error Handling**

* **Java Check:** Verify java \-version before execution.  
* **Path Verification:** Ensure all file paths are absolute or correctly relative.