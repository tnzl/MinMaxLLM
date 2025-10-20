#include <tensor/safetensors.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <safetensor_file>\n";
        return 1;
    }

    try
    {
        Safetensor st(argv[1]);

        std::cout << "Keys in the safetensor file:\n";
        for (const auto &key : st.keys())
        {
            const TensorInfo *info = st.getTensorInfo(key);
            if (info)
            {
                std::cout << "Key: " << key << "\n";
                std::cout << "  Dtype: " << info->dtype << "\n";
                std::cout << "  Shape: [";
                for (size_t i = 0; i < info->shape.size(); ++i)
                {
                    std::cout << info->shape[i];
                    if (i + 1 < info->shape.size())
                        std::cout << ", ";
                }
                std::cout << "]\n";
                std::cout << "  Byte Size: " << st.tensorByteSize(key) << "\n\n";
            }
        }

        std::cout << "Metadata:\n";
        st.printHeader();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}