from normalize import (
    normalize_name,
    normalize_name_ascii,
    normalize_address,
    normalize_address_ascii,
    extract_numbers
)


examples = [
    "Payne Énterprises",
    "Lumay Bóral Inc.",
    "Hendricks and  Flowers Inc",
    "Maure Williams Colombier Inc",
    "राज Investments LLP",
]


print("\nNAME NORMALIZATION")
print("=" * 60)

for name in examples:
    print("Original :", name)
    print("Normal   :", normalize_name(name))
    print("ASCII    :", normalize_name_ascii(name))
    print()


addresses = [
    "3315 Fremont Street, Peoria, IL",
    "1056-1060 BELDEN AVE, PO BOX 8807, AKRON, OH",
    "630 45th Terrace, Kansas City, MO",
    "Af-684, Nandgram Near Mother India Public School"
]


print("\nADDRESS NORMALIZATION")
print("=" * 60)

for address in addresses:
    print("Original :", address)
    print("Normal   :", normalize_address(address))
    print("ASCII    :", normalize_address_ascii(address))
    print("Numbers  :", extract_numbers(address))
    print()