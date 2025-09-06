# scripts/transaction_analyzer.py

import pandas as pd
import os

class TransactionAnalyzer:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.transactions = pd.DataFrame()
        if file_path and os.path.exists(file_path):
            try:
                self.load_data(file_path)
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not auto-load transaction data on init: {e}")
        elif file_path:
            print(f"Warning: Transaction file not found at '{file_path}' during initialization.")

    def load_data(self, file_path):
        """Loads transaction data from a CSV or Excel file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Transaction data file not found at: {file_path}")
        
        self.file_path = file_path # Update file path if loaded dynamically

        if file_path.endswith('.csv'):
            self.transactions = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            self.transactions = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
        
        self._preprocess_data()
        print(f"Loaded {len(self.transactions)} transactions from {os.path.basename(file_path)}")

    def _preprocess_data(self):
        """Cleans and prepares the loaded transaction data."""
        # Standardize column names (case-insensitive)
        self.transactions.columns = [col.strip().replace(' ', '_').lower() for col in self.transactions.columns]

        # Rename columns to expected names if they differ
        # Assuming you've provided: Date, Transaction ID, Name, Type, Amount
        # Let's map them to a consistent internal naming for easier access
        column_mapping = {
            'date': 'transaction_date',
            'transaction_id': 'transaction_id',
            'name': 'party_name',
            'type': 'transaction_type',
            'amount': 'amount'
        }
        self.transactions = self.transactions.rename(columns=column_mapping)
        
        # Ensure 'transaction_date' is datetime
        if 'transaction_date' in self.transactions.columns:
            self.transactions['transaction_date'] = pd.to_datetime(self.transactions['transaction_date'], errors='coerce')
            self.transactions.dropna(subset=['transaction_date'], inplace=True) # Drop rows with invalid dates

        # Ensure 'amount' is numeric
        if 'amount' in self.transactions.columns:
            self.transactions['amount'] = pd.to_numeric(self.transactions['amount'], errors='coerce')
            self.transactions.dropna(subset=['amount'], inplace=True) # Drop rows with invalid amounts

        # Standardize 'transaction_type' (e.g., 'debit'/'credit' to 'Purchase'/'Sale')
        if 'transaction_type' in self.transactions.columns:
            self.transactions['transaction_type'] = self.transactions['transaction_type'].str.lower().str.strip()
            self.transactions['transaction_type'] = self.transactions['transaction_type'].replace({
                'credit': 'Sale',
                'debit': 'Purchase'
            })
            # Filter out types not explicitly 'Sale' or 'Purchase' if desired, or keep them
            # For now, we'll assume other types are just 'Other'
            self.transactions['transaction_type'] = self.transactions['transaction_type'].apply(
                lambda x: x if x in ['Sale', 'Purchase'] else 'Other'
            )

        print("Transaction data preprocessed successfully.")

    def get_summary(self):
        """Generates a high-level summary of transactions."""
        if self.transactions.empty:
            return "No transaction data available for summary."

        total_sales = self.transactions[self.transactions['transaction_type'] == 'Sale']['amount'].sum()
        total_purchases = self.transactions[self.transactions['transaction_type'] == 'Purchase']['amount'].sum()
        
        summary_str = (
            f"Transaction Summary:\n"
            f"Total Sales: INR {total_sales:,.2f}\n"
            f"Total Purchases: INR {total_purchases:,.2f}\n"
            f"Number of Sales Transactions: {len(self.transactions[self.transactions['transaction_type'] == 'Sale'])}\n"
            f"Number of Purchase Transactions: {len(self.transactions[self.transactions['transaction_type'] == 'Purchase'])}\n"
        )
        return summary_str

    def find_potential_itc_opportunities(self):
        """
        Identifies simple potential ITC opportunities.
        This is a placeholder and needs to be expanded with real GST logic.
        """
        if self.transactions.empty:
            return "No transaction data available to identify ITC opportunities."
        
        # Simple logic: assume all purchases are potentially eligible for ITC
        # In a real scenario, you'd check GSTINs, invoice details, nature of goods/services etc.
        potential_itc_transactions = self.transactions[
            (self.transactions['transaction_type'] == 'Purchase') & 
            (self.transactions['amount'] > 0) # Only positive purchase amounts
        ]
        
        if potential_itc_transactions.empty:
            return "No obvious purchase transactions found for potential ITC."

        itc_opportunities = []
        for _, row in potential_itc_transactions.head(5).iterrows(): # Show top 5 for brevity
            itc_opportunities.append(
                f"- Purchase from '{row['party_name']}' on {row['transaction_date'].strftime('%Y-%m-%d')} for INR {row['amount']:,.2f} "
                f"(Transaction ID: {row['transaction_id']}). This might be eligible for ITC."
            )
        
        if len(potential_itc_transactions) > 5:
            itc_opportunities.append(f"... and {len(potential_itc_transactions) - 5} more potential ITC transactions.")
        
        return "Potential Input Tax Credit (ITC) Opportunities (simplified):\n" + "\n".join(itc_opportunities)

    # You can add more analysis methods here:
    # - `check_threshold_compliance()` for composition levy
    # - `identify_late_filing_risks()` based on dates
    # - `categorize_expenses_for_gstr1()` etc.

# Example usage (for testing this module directly)
if __name__ == "__main__":
    # Assuming transactions.csv is in the root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    transaction_file_path = os.path.join(current_dir, '..', 'transactions.csv')

    try:
        analyzer = TransactionAnalyzer(transaction_file_path)
        print("\n--- Summary ---")
        print(analyzer.get_summary())
        print("\n--- ITC Opportunities ---")
        print(analyzer.find_potential_itc_opportunities())
        
        # Example of loading another file
        # print("\n--- Loading another file (example) ---")
        # excel_file_path = os.path.join(current_dir, '..', 'another_transactions.xlsx')
        # # Create a dummy excel file for testing:
        # pd.DataFrame({
        #     'Date': ['2023-11-01'], 'Transaction ID': ['T201'], 'Name': ['Vendor Y'],
        #     'Type': ['Debit'], 'Amount': [3000.00]
        # }).to_excel(excel_file_path, index=False)
        # 
        # analyzer.load_data(excel_file_path)
        # print(analyzer.get_summary())
        # os.remove(excel_file_path) # Clean up dummy file

    except FileNotFoundError as e:
        print(e)
        print("Please make sure 'transactions.csv' exists in the project root directory.")
    except ValueError as e:
        print(e)