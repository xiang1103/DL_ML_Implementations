class Node():
    def __init__(self, data):
        self.val= data 
        self.next= None  

class LinkedList:
    def __init__(self):
        self.head= None 
    def addFirst(self,data):
        new_node= Node(data)
        new_node.next= self.head
        self.head= new_node 
    def addLast(self,data):
        new_node= Node(data)
        if self.head is None: 
            self.addFirst(data)
            return
        else:
            cur= self.head 
            while cur.next: 
                cur= cur.next 
            cur.next= new_node 
    def insertIndex(self,data, index):
        if index==0:
            self.addFirst(data)
        cur= self.head 
        pos= 0
        while(cur is not None and pos<index-1):
            cur= cur.next 
            pos+=1 
        if cur is not None: 
            new_node= Node(data)
            new_node.next= cur.next 
            cur.next= new_node
        else:
            print ("InsertIndex index out of range")
    def updateNode(self,data, index):
        cur = self.head 
        pos= 0 
        while (cur is not None and pos<index):
            cur= cur.next 
            pos +=1 
        if cur is not None: 
            cur.val= data 
        else: 
            print ("updateNode index out of range")
    def removeHead(self):
        if self.head is None:
            print ("RemoveHead but head is None")
            return
        else: 
            self.head= self.head.next 
    def removeLast(self):
        if self.head is None: 
            print ("removeLast but head is None")
            return 
        if self.head.next is None: 
            self.head=None 
            return 
        cur = self.head 
        while cur.next.next:
            cur= cur.next 
        cur.next=None
    def removeIndex(self, index):
        if index==0:
            self.removeHead()
            return 
        cur= self.head 
        pos =0
        while (cur is not None and pos<index-1):
            cur= cur.next 
            pos+=1 
        if (cur is not None and cur.next is not None):
            cur.next= cur.next.next
        else: 
            print ("removeIndex out of range")
    def print(self):
        cur= self.head 
        if cur is None: 
            print("Empty list")
            return
        while (cur):
            print (cur.val, end= '->')
            cur=cur.next
    


def RemoveDups (self):  #self is a linkedlist 
    cur= self.head 
    if cur is None: #corner case, when there are no nodes 
        return self 
    '''
    METHOD 1: solve with hashtable (store the val)
    '''
    val_table= {} 
    val_table[cur.val] = 1
    while cur.next is not None: 
        val_table[cur.val]= 1 
        if cur.next.val in val_table: 
            cur.next= cur.next.next 
        else:
            cur= cur.next
    return self    #base condition if we don't fail in the while loop, also handles the case when we only have 1 node

        
