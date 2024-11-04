class CollaborativeFilteringMatrix:
    def __init__(self, num_users, num_products):
        self.matrix = [[0] * num_products for _ in range(num_users)]

    def update_interaction(self, user_id, product_id, rating):
        self.matrix[user_id][product_id] = rating


class LinkedListNode:
    def __init__(self, product_id):
        self.product_id = product_id
        self.next = None

class UserSessionMap:
    def __init__(self):
        self.sessions = {}

    def add_interaction(self, user_id, product_id):
        if user_id not in self.sessions:
            self.sessions[user_id] = LinkedListNode(product_id)
        else:
            new_node = LinkedListNode(product_id)
            new_node.next = self.sessions[user_id]
            self.sessions[user_id] = new_node

class KDTreeNode:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

class KDTree:
    def __init__(self):
        self.root = None

    def insert(self, root, point, depth=0):
        if root is None:
            return KDTreeNode(point)
        k = len(point)
        axis = depth % k
        if point[axis] < root.point[axis]:
            root.left = self.insert(root.left, point, depth + 1)
        else:
            root.right = self.insert(root.right, point, depth + 1)
        return root
