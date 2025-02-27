import json

inventory = {}
unique_isbns = set()

while True:
    print("\nBookstore Management System")
    print("1. Add Book")
    print("2. Remove Book")
    print("3. Update Book")
    print("4. Display Inventory")
    print("5. Purchase Book")
    print("6. Exit")
    
    choice = input("Enter your choice: ")
    
    if choice == "1":
        isbn = input("Enter ISBN: ")
        title = input("Enter Title: ")
        author = input("Enter Author: ")
        price = float(input("Enter Price: "))
        quantity = int(input("Enter Quantity: "))
        
        if isbn not in inventory:
            inventory[isbn] = {"title": title, "author": author, "price": price, "quantity": quantity}
            unique_isbns.add(isbn)
            print("Book added successfully!")
        else:
            print("Book already exists. Use update option.")
    
    elif choice == "2":
        isbn = input("Enter ISBN to remove: ")
        
        if isbn in inventory:
            del inventory[isbn]
            unique_isbns.remove(isbn)
            print("Book removed successfully!")
        else:
            print("Book not found.")
    
    elif choice == "3":
        isbn = input("Enter ISBN: ")
        attribute = input("Enter attribute to update (title, author, price, quantity): ")
        value = input("Enter new value: ")
        
        if isbn in inventory and attribute in inventory[isbn]:
            if attribute == "price":
                inventory[isbn][attribute] = float(value)
            elif attribute == "quantity":
                inventory[isbn][attribute] = int(value)
            else:
                inventory[isbn][attribute] = value
            print("Book updated successfully!")
        else:
            print("Invalid ISBN or attribute.")
    
    elif choice == "4":
        if not inventory:
            print("No books in inventory.")
        else:
            print(json.dumps(inventory, indent=4))
    
    elif choice == "5":
        isbn = input("Enter ISBN: ")
        quantity = int(input("Enter quantity to purchase: "))
        
        if isbn in inventory and inventory[isbn]["quantity"] >= quantity:
            inventory[isbn]["quantity"] -= quantity
            print(f"Purchase successful! {quantity} copies of '{inventory[isbn]['title']}' bought.")
        else:
            print("Book not found or insufficient stock.")
    
    elif choice == "6":
        print("Exiting...")
        break
    
    else:
        print("Invalid choice. Please try again.")
