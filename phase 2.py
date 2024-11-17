# Collaborative Filtering Matrix Implementation
class CollaborativeFilteringMatrix:
    def __init__(self, num_users, num_products):
        # Initialize a user-product interaction matrix
        self.matrix = [[0] * num_products for _ in range(num_users)]

    def update_interaction(self, user_id, product_id, rating):
        """Update the matrix with a user's product rating."""
        self.matrix[user_id][product_id] = rating

    def get_user_ratings(self, user_id):
        """Retrieve all ratings given by a specific user."""
        return self.matrix[user_id]

    def get_product_ratings(self, product_id):
        """Retrieve all ratings for a specific product."""
        return [row[product_id] for row in self.matrix]

# Linked List Implementation for User Session
class LinkedListNode:
    def __init__(self, product_id):
        self.product_id = product_id
        self.next = None

class UserSessionMap:
    def __init__(self):
        self.sessions = {}

    def add_interaction(self, user_id, product_id):
        """Add a product interaction for a specific user."""
        if user_id not in self.sessions:
            self.sessions[user_id] = LinkedListNode(product_id)
        else:
            new_node = LinkedListNode(product_id)
            new_node.next = self.sessions[user_id]
            self.sessions[user_id] = new_node

    def get_user_session(self, user_id):
        """Retrieve all products interacted with by a specific user."""
        products = []
        current = self.sessions.get(user_id, None)
        while current:
            products.append(current.product_id)
            current = current.next
        return products

# KDTree Implementation for Product Similarity
class KDTreeNode:
    def __init__(self, point, left=None, right=None, product_id=None):
        self.point = point
        self.left = left
        self.right = right
        self.product_id = product_id  # Added for product association

class KDTree:
    def __init__(self):
        self.root = None

    def insert(self, root, point, product_id=None, depth=0):
        """Insert a point into the KDTree."""
        if root is None:
            return KDTreeNode(point, product_id=product_id)
        k = len(point)
        axis = depth % k
        if point[axis] < root.point[axis]:
            root.left = self.insert(root.left, point, product_id, depth + 1)
        else:
            root.right = self.insert(root.right, point, product_id, depth + 1)
        return root

    def add_product(self, point, product_id):
        """Add a product's feature vector to the tree."""
        self.root = self.insert(self.root, point, product_id)

    def nearest_neighbor(self, root, target, depth=0, best=None):
        """Find the nearest neighbor for a given point."""
        if root is None:
            return best

        k = len(target)
        axis = depth % k

        # Check the current node
        if best is None or self.distance(root.point, target) < self.distance(best.point, target):
            best = root

        # Recur down the tree
        if target[axis] < root.point[axis]:
            best = self.nearest_neighbor(root.left, target, depth + 1, best)
        else:
            best = self.nearest_neighbor(root.right, target, depth + 1, best)

        # Check the other branch
        if abs(target[axis] - root.point[axis]) < self.distance(best.point, target):
            if target[axis] < root.point[axis]:
                best = self.nearest_neighbor(root.right, target, depth + 1, best)
            else:
                best = self.nearest_neighbor(root.left, target, depth + 1, best)

        return best

    @staticmethod
    def distance(point1, point2):
        """Calculate Euclidean distance between two points."""
        return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5


if __name__ == "__main__":
    # Collaborative Filtering Example
    cf_matrix = CollaborativeFilteringMatrix(num_users=5, num_products=5)
    cf_matrix.update_interaction(0, 0, 5)  # User 0 rated Product 0 as 5
    cf_matrix.update_interaction(1, 2, 3)  # User 1 rated Product 2 as 3
    print("User 0 Ratings:", cf_matrix.get_user_ratings(0))
    print("Product 2 Ratings:", cf_matrix.get_product_ratings(2))

    # User Session Example
    user_sessions = UserSessionMap()
    user_sessions.add_interaction(0, 101)
    user_sessions.add_interaction(0, 102)
    user_sessions.add_interaction(1, 103)
    print("User 0 Session:", user_sessions.get_user_session(0))
    print("User 1 Session:", user_sessions.get_user_session(1))

    # KDTree Example
    kdtree = KDTree()
    kdtree.add_product([1.0, 2.0], product_id=201)
    kdtree.add_product([3.0, 4.0], product_id=202)
    kdtree.add_product([5.0, 6.0], product_id=203)
    target_point = [2.0, 3.0]
    nearest = kdtree.nearest_neighbor(kdtree.root, target_point)
    print("Nearest Product to", target_point, "is Product ID:", nearest.product_id)

