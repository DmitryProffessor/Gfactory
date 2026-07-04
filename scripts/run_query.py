import argparse
import json
from src.pipeline import HypothesisGenerator
from src.schemas import QueryRequest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", required=True, help="Path to FAISS index")
    parser.add_argument("--query_file", required=True, help="JSON file with query parameters")
    parser.add_argument("--output", required=True, help="Output JSON file for hypotheses")
    args = parser.parse_args()

    with open(args.query_file, 'r') as f:
        query_dict = json.load(f)
    request = QueryRequest(**query_dict)

    generator = HypothesisGenerator(index_path=args.index_path)
    response = generator.generate(request)

    with open(args.output, 'w') as f:
        json.dump(response.dict(), f, indent=2)
    print(f"Hypotheses written to {args.output}")

if __name__ == "__main__":
    main()
