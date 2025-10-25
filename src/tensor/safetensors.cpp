#include <tensor/safetensors.h>

// ============================================================================
// MiniJson Implementation
// ============================================================================

MiniJson::MiniJson(const char *headerData, size_t size)
{
    std::string json(headerData, size);
    parse(json);
}

const TensorInfo *MiniJson::get(const std::string &key) const
{
    auto it = tensors.find(key);
    return (it != tensors.end()) ? &it->second : nullptr;
}

void MiniJson::printMetadata() const
{
    if (metadata.empty())
    {
        std::cout << "(no __metadata__ present)\n";
        return;
    }
    std::cout << "__metadata__:\n";
    for (const auto &[k, v] : metadata)
    {
        std::cout << "  " << k << ": " << v << "\n";
    }
    std::cout << "\n";
}

void MiniJson::print() const
{
    printMetadata();
    for (const auto &k : key_order)
    {
        const auto &t = tensors.at(k);
        std::cout << "Tensor: " << k << "\n"
                  << "  dtype: " << t.dtype << "\n"
                  << "  shape: [";
        for (size_t i = 0; i < t.shape.size(); ++i)
        {
            std::cout << t.shape[i];
            if (i + 1 < t.shape.size())
                std::cout << ", ";
        }
        std::cout << "]\n"
                  << "  offsets: (" << t.data_offsets.first << ", " << t.data_offsets.second << ")\n\n";
    }
}

void MiniJson::skipSpaces(std::istringstream &ss)
{
    while (ss && std::isspace(ss.peek()))
        ss.get();
}

std::string MiniJson::readQuoted(std::istringstream &ss)
{
    skipSpaces(ss);
    if (ss.get() != '"')
        return "";
    std::string s;
    char c;
    while (ss.get(c))
    {
        if (c == '"')
            break;
        s += c;
    }
    return s;
}

std::vector<size_t> MiniJson::readArray(std::istringstream &ss)
{
    std::vector<size_t> arr;
    skipSpaces(ss);
    if (ss.get() != '[')
        return arr;
    skipSpaces(ss);
    std::string num;
    char c;
    while (ss.get(c))
    {
        if (c == ']')
        {
            if (!num.empty())
                arr.push_back(std::stoull(num));
            break;
        }
        if (std::isdigit(c))
            num += c;
        else if (c == ',')
        {
            if (!num.empty())
            {
                arr.push_back(std::stoull(num));
                num.clear();
            }
        }
    }
    return arr;
}

std::pair<size_t, size_t> MiniJson::readPair(std::istringstream &ss)
{
    auto arr = readArray(ss);
    if (arr.size() == 2)
        return {arr[0], arr[1]};
    return {0, 0};
}

std::unordered_map<std::string, std::string> MiniJson::readStringMap(std::istringstream &ss)
{
    std::unordered_map<std::string, std::string> result;
    skipSpaces(ss);
    if (ss.get() != '{')
        return result;

    while (true)
    {
        skipSpaces(ss);
        if (ss.peek() == '}')
        {
            ss.get();
            break;
        }

        std::string key = readQuoted(ss);
        skipSpaces(ss);
        if (ss.get() != ':')
            break;
        skipSpaces(ss);
        std::string val = readQuoted(ss);
        result[key] = val;

        skipSpaces(ss);
        if (ss.peek() == ',')
        {
            ss.get();
            continue;
        }
        else if (ss.peek() == '}')
        {
            ss.get();
            break;
        }
    }
    return result;
}

void MiniJson::parse(const std::string &json)
{
    std::istringstream ss(json);
    skipSpaces(ss);
    if (ss.get() != '{')
        return;

    while (true)
    {
        skipSpaces(ss);
        if (ss.peek() == '}')
        {
            ss.get();
            break;
        }

        std::string key = readQuoted(ss);
        skipSpaces(ss);
        if (ss.get() != ':')
            break;
        skipSpaces(ss);

        // Handle metadata block separately
        if (key == "__metadata__")
        {
            metadata = readStringMap(ss);
        }
        else
        {
            if (ss.get() != '{')
                break;
            TensorInfo info;
            while (true)
            {
                std::string field = readQuoted(ss);
                skipSpaces(ss);
                if (ss.get() != ':')
                    break;
                skipSpaces(ss);

                if (field == "dtype")
                {
                    info.dtype = readQuoted(ss);
                }
                else if (field == "shape")
                {
                    info.shape = readArray(ss);
                }
                else if (field == "data_offsets")
                {
                    info.data_offsets = readPair(ss);
                }

                skipSpaces(ss);
                if (ss.peek() == ',')
                {
                    ss.get();
                    continue;
                }
                else if (ss.peek() == '}')
                {
                    ss.get();
                    break;
                }
            }

            tensors[key] = info;
            key_order.push_back(key);
        }

        skipSpaces(ss);
        if (ss.peek() == ',')
        {
            ss.get();
            continue;
        }
        else if (ss.peek() == '}')
        {
            ss.get();
            break;
        }
    }
}

// ============================================================================
// Safetensor Implementation
// ============================================================================

Safetensor::Safetensor(const std::string &path, bool mmap)
    : data(nullptr),
      data_size(0),
      is_mmap(mmap),
      hFile_(INVALID_HANDLE_VALUE),
      hMap_(nullptr)
{
    load(path);
}

Safetensor::~Safetensor()
{
    if (is_mmap)
    {
        cleanup_mmap();
    }
    else
    {
        delete[] data;
    }
}

void Safetensor::cleanup_mmap()
{
    if (data)
        UnmapViewOfFile(data);
    data = nullptr;
    if (hMap_)
        CloseHandle(hMap_);
    hMap_ = nullptr;
    if (hFile_ != INVALID_HANDLE_VALUE)
        CloseHandle(hFile_);
    hFile_ = INVALID_HANDLE_VALUE;
}

// Prefetch wrapper (your safe version)
bool Safetensor::windows_advise(void *ptr, size_t size) noexcept
{
    if (!ptr || size == 0)
        return false;

    using PrefetchVirtualMemoryFn = BOOL(WINAPI *)(HANDLE, ULONG, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static auto fn = reinterpret_cast<PrefetchVirtualMemoryFn>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory"));
    if (!fn)
        return false; // Not supported

    WIN32_MEMORY_RANGE_ENTRY range;
    range.VirtualAddress = ptr;
    range.NumberOfBytes = size;

    return fn(GetCurrentProcess(), 1, &range, 0) != 0;
}

size_t Safetensor::tensorByteSize(const std::string &key) const
{
    const auto *info = json.get(key);
    if (!info)
        return 0;
    return info->data_offsets.second - info->data_offsets.first;
}

void Safetensor::load(const std::string &path)
{
    if (is_mmap)
    {
        load_mmap(path);
    }
    else
    {
        load_memory(path);
    }
}

void Safetensor::load_mmap(const std::string &path)
{
    hFile_ = INVALID_HANDLE_VALUE;
    hMap_ = nullptr;

    hFile_ = CreateFileW(std::wstring(path.begin(), path.end()).c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile_ == INVALID_HANDLE_VALUE)
        throw std::runtime_error("Cannot open file: " + path);

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile_, &fileSize))
    {
        CloseHandle(hFile_);
        throw std::runtime_error("Cannot get file size: " + path);
    }

    size_t file_size = static_cast<size_t>(fileSize.QuadPart);

    hMap_ = CreateFileMappingW(hFile_, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap_)
    {
        CloseHandle(hFile_);
        throw std::runtime_error("Cannot create file mapping: " + path);
    }

    uint8_t *file_data = static_cast<uint8_t *>(MapViewOfFile(hMap_, FILE_MAP_READ, 0, 0, 0));
    if (!file_data)
    {
        CloseHandle(hMap_);
        CloseHandle(hFile_);
        throw std::runtime_error("Cannot map view of file: " + path);
    }
    // Read header length (first 8 bytes, little endian)
    uint64_t header_size = *reinterpret_cast<uint64_t *>(file_data);

    // Parse header using MiniJson
    json = MiniJson(reinterpret_cast<char *>(file_data + sizeof(uint64_t)), header_size);

    data = file_data + sizeof(uint64_t) + header_size;
    data_size = file_size - (sizeof(uint64_t) + header_size);
}

void Safetensor::load_memory(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    // Read header length (first 8 bytes, little endian)
    uint64_t header_size = 0;
    f.read(reinterpret_cast<char *>(&header_size), sizeof(uint64_t));
    if (!f)
        throw std::runtime_error("Failed to read header size");

    // Read header JSON
    std::vector<char> headerBuf(header_size);
    f.read(headerBuf.data(), header_size);
    if (!f)
        throw std::runtime_error("Failed to read header JSON");

    // Parse header using MiniJson
    json = MiniJson(headerBuf.data(), header_size);

    // Read remaining tensor data
    f.seekg(0, std::ios::end);
    size_t total_size = static_cast<size_t>(f.tellg());
    data_size = total_size - (sizeof(uint64_t) + header_size);
    // Allocate memory for tensor data of data_size
    data = new uint8_t[data_size];

    f.seekg(sizeof(uint64_t) + header_size, std::ios::beg);
    f.read(reinterpret_cast<char *>(data), data_size);
    if (!f)
        throw std::runtime_error("Failed to read tensor data");

    f.close();
}