#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <cstddef>
#include <cstdint>

struct TensorInfo
{
    std::string dtype;
    std::vector<size_t> shape;
    std::pair<size_t, size_t> data_offsets;
};

class MiniJson
{
public:
    MiniJson(const char *headerData, size_t size);
    MiniJson() = default;

    const std::vector<std::string> &keys() const { return key_order; }
    const TensorInfo *get(const std::string &key) const;
    const std::unordered_map<std::string, std::string> &getMetadata() const { return metadata; }

    void printMetadata() const;
    void print() const;

private:
    std::unordered_map<std::string, TensorInfo> tensors;
    std::unordered_map<std::string, std::string> metadata;
    std::vector<std::string> key_order;

    // Utility parsing functions
    static void skipSpaces(std::istringstream &ss);
    static std::string readQuoted(std::istringstream &ss);
    static std::vector<size_t> readArray(std::istringstream &ss);
    static std::pair<size_t, size_t> readPair(std::istringstream &ss);
    static std::unordered_map<std::string, std::string> readStringMap(std::istringstream &ss);

    void parse(const std::string &json);
};

class SafeTensor
{
public:
    SafeTensor(const std::string &path);

    // Accessors
    const std::vector<std::string> &keys() const { return json.keys(); }
    const TensorInfo *getTensorInfo(const std::string &name) const { return json.get(name); }
    const std::unordered_map<std::string, std::string> &getMetadata() const { return json.getMetadata(); }

    void printHeader() const { json.print(); }

    // Returns pointer to tensor data (not owning)
    const uint8_t *tensorDataPtr(const std::string &key) const;
    size_t tensorByteSize(const std::string &key) const;

private:
    MiniJson json;
    std::vector<uint8_t> data;

    void load(const std::string &path);
};