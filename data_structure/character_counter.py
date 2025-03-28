''' 
Given a string, return a table of all the characters and their frequency counter  (will be all lower case and white space removed)
return a sorted by descending order by the frequency 
'''

def character_counter(s:str): 
    hash= {} 
    s= s.lower().replace (" ","")
    # loop through the entire string 
    for c in s: 
        if c in hash: 
            hash[c] +=1 
        else: 
            hash[c]=1 
    print("Total Length without white spaces:", len(s))
    return dict(sorted(hash.items(),key=lambda item:item[1], reverse=True))


def main(): 
    input= "Build Your Career brick by brick"
    print(character_counter(input))
    print(len(input))

if __name__ == "__main__":
    main() 